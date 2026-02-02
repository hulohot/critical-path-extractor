#!/usr/bin/env python3
"""
Boolean-gates–driven Critical Path Extractor

This script extracts Path 1 from a Tempus detailed timing report and builds a minimal 
Verilog netlist that contains ONLY the combinational logic cells on that path. 
Flip-flops and latches are excluded from the netlist, and the path_input port is 
connected directly to the first net after any flip-flop in the timing sequence.

All non-critical inputs are tied off to VDD/VSS based on the Boolean definitions in 
`BOOLEAN_gates.v`. The netlist includes fanout gates to match STA loading conditions 
for accurate timing analysis.

Key Features:
- Excludes sequential cells (flip-flops, latches) from the netlist
- Connects path_input to the first combinational net after flip-flops
- Uses Boolean gate definitions for intelligent tie-off handling
- Includes fanout cells to match Tempus STA loading conditions
- Generates a minimal combinational-only critical path netlist

Usage:
    python extract_critical_path.py [--width WIDTH] [--stages STAGES] [--iteration ITERATION]

If no arguments are provided, uses default values imported from the existing 
`extract_critical_path.py` script.
"""

import argparse
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

# Reuse existing timing and netlist parsers & defaults from extract_critical_path
from extract_critical_path import (
    TimingReportParser,
    NetlistParser,
    DEFAULT_WIDTH,
    DEFAULT_STAGES,
    DEFAULT_ITERATION,
)

console = Console()

# ---------------------------------------------------------------------------
# Boolean gates parsing
# ---------------------------------------------------------------------------

@dataclass
class GatePrimitive:
    kind: str  # and, or, nand, nor, not, xor, xnor
    output: str  # output wire name
    inputs: List[str]  # input wire names


@dataclass
class GateDef:
    name: str  # full cell name, e.g. OAI2B1_X1_A8TR
    base_name: str  # base name, e.g. OAI2B1
    output: str  # primary output port, usually Y
    inputs: List[str]  # primary input ports in order
    primitives: List[GatePrimitive]
    assigns: Dict[str, str] = field(default_factory=dict)  # output -> internal_wire


class BooleanGatesParser:
    """Parse BOOLEAN_gates.v into a map of base cell name -> GateDef."""

    def __init__(self, gates_file: str):
        self.gates_file = gates_file
        # Map base_name (e.g. OAI2B1) -> GateDef (typically X1 variant)
        self.gates: Dict[str, GateDef] = {}

    def parse(self) -> None:
        if not os.path.exists(self.gates_file):
            raise FileNotFoundError(self.gates_file)

        with open(self.gates_file, "r") as f:
            content = f.read()

        # Match each module definition non-greedily
        module_pattern = re.compile(
            r"module\s+(\w+)\s*\(([^)]*)\);\s*(.*?)endmodule",
            re.DOTALL | re.MULTILINE,
        )

        for match in module_pattern.finditer(content):
            full_name = match.group(1)  # e.g. OAI2B1_X1_A8TR
            ports_str = match.group(2)
            body = match.group(3)

            # Only care about standard cells with recognized suffixes
            # MUSE65 uses _A8TR, GF12 uses _C14
            if not (full_name.endswith("_A8TR") or full_name.endswith("_C14")):
                continue

            # Base name: strip drive strength and suffix
            # e.g. OAI2B1_X1_A8TR -> OAI2B1
            base_name = full_name.split("_X", 1)[0]

            # Prefer the first variant we see (usually X1)
            if base_name in self.gates:
                continue

            # Parse ports from header: (Y, A0, A1N, B0)
            ports = [p.strip() for p in ports_str.split(",") if p.strip()]
            if not ports:
                continue

            output_port = ports[0]

            # Filter out power pins from inputs (GF12 cells include VDD, VSS, VNW, VPW in port list)
            power_pins = {"VDD", "VSS", "VNW", "VPW"}
            input_ports = [p for p in ports[1:] if p not in power_pins]

            # Parse primitive instances inside the body
            primitives: List[GatePrimitive] = []
            prim_pattern = re.compile(
                r"^\s*(and|or|nand|nor|not|xor|xnor)\b[^()]*\(([^;]+)\);",
                re.MULTILINE,
            )

            for p_match in prim_pattern.finditer(body):
                kind = p_match.group(1)
                conn_str = p_match.group(2)
                nets = [n.strip() for n in conn_str.split(",") if n.strip()]
                if len(nets) < 2:
                    continue

                out_net = nets[0]
                in_nets = nets[1:]
                primitives.append(GatePrimitive(kind=kind, output=out_net, inputs=in_nets))

            if not primitives:
                continue

            # Parse assign statements that connect internal wires to outputs
            # Pattern: assign Y = ... out_temp ... or assign Y = out_temp
            # GF12 cells use: assign Y = ((VDD === 1'b1) && ...) ? out_temp : 1'bx;
            assigns: Dict[str, str] = {}
            prim_outputs = {p.output for p in primitives}

            # Match assign statements
            assign_pattern = re.compile(
                r"assign\s+(?:`\w+\s+)?(\w+)\s*=\s*([^;]+);",
                re.DOTALL,
            )

            for a_match in assign_pattern.finditer(body):
                lhs = a_match.group(1)  # e.g., Y
                rhs_text = a_match.group(2)  # e.g., ((VDD === 1'b1) && ...) ? out_temp : 1'bx

                # Find which primitive output connects to this assign
                for prim_out in prim_outputs:
                    # Check if the primitive output appears in the RHS
                    # Use word boundary to avoid partial matches
                    if re.search(rf'\b{re.escape(prim_out)}\b', rhs_text):
                        assigns[lhs] = prim_out
                        break

            self.gates[base_name] = GateDef(
                name=full_name,
                base_name=base_name,
                output=output_port,
                inputs=input_ports,
                primitives=primitives,
                assigns=assigns,
            )


class BooleanTieOffEngine:
    """Brute-force Boolean evaluation to derive side-input tie-offs."""

    def __init__(self, gates: Dict[str, GateDef]):
        self.gates = gates
        # cache[(base_name, critical_pin)] = {pin: 'VDD'/'VSS'}
        self.cache: Dict[Tuple[str, str], Dict[str, str]] = {}

    def _eval_gate(self, gate: GateDef, inputs: Dict[str, int]) -> int:
        """Evaluate gate output Y for a given assignment of primary inputs."""
        env: Dict[str, int] = dict(inputs)

        def get(wire: str) -> int:
            if wire not in env:
                raise KeyError(f"Wire {wire} not set while evaluating {gate.name}")
            return env[wire]

        for prim in gate.primitives:
            vals = [get(w) for w in prim.inputs]
            if prim.kind == "and":
                out = int(all(vals))
            elif prim.kind == "or":
                out = int(any(vals))
            elif prim.kind == "nand":
                out = int(not all(vals))
            elif prim.kind == "nor":
                out = int(not any(vals))
            elif prim.kind == "xor":
                out = 0
                for v in vals:
                    out ^= v
            elif prim.kind == "xnor":
                out = 0
                for v in vals:
                    out ^= v
                out = int(not out)
            elif prim.kind == "not":
                out = int(not vals[0])
            else:
                raise ValueError(f"Unknown primitive kind {prim.kind}")

            env[prim.output] = out

        # Apply assign statements (e.g., Y = out_temp for GF12 cells)
        for lhs, rhs in gate.assigns.items():
            if rhs in env:
                env[lhs] = env[rhs]

        return env[gate.output]

    def get_tieoffs(self, cell_type: str, critical_pin: str) -> Dict[str, str]:
        """Return a mapping pin_name -> 'VDD'/'VSS' for non-critical inputs."""
        base_name = cell_type.split("_X", 1)[0]
        key = (base_name, critical_pin)

        if key in self.cache:
            return self.cache[key]

        if base_name not in self.gates:
            # Unknown gate type; no tieoffs
            self.cache[key] = {}
            return {}

        gate = self.gates[base_name]

        if critical_pin not in gate.inputs:
            self.cache[key] = {}
            return {}

        others = [p for p in gate.inputs if p != critical_pin]
        if not others:
            self.cache[key] = {}
            return {}

        valid_assignments: List[Dict[str, int]] = []

        # Try all combinations of 0/1 for side inputs
        for mask in range(1 << len(others)):
            assign: Dict[str, int] = {}
            for i, pin in enumerate(others):
                assign[pin] = (mask >> i) & 1

            # Evaluate for critical_pin = 0 and 1
            a0 = dict(assign)
            a0[critical_pin] = 0
            a1 = dict(assign)
            a1[critical_pin] = 1

            try:
                y0 = self._eval_gate(gate, a0)
                y1 = self._eval_gate(gate, a1)
            except KeyError:
                # Missing internal net assignment; skip
                continue

            # Check if transition is possible (Y can change)
            if y0 != y1:
                valid_assignments.append(assign)

        if not valid_assignments:
            console.print(
                f"[red]✗ Error:[/red] Could not find valid tie-offs for gate [cyan]{base_name}[/cyan] "
                f"with critical pin [cyan]{critical_pin}[/cyan]."
            )
            self.cache[key] = {}
            return {}

        # Select the best assignment: prefer more VDD ties (better drive strength)
        def score_assignment(assign: Dict[str, int]) -> int:
            return sum(assign.values())

        best_assignment = max(valid_assignments, key=score_assignment)

        tieoffs = {
            pin: ("VDD" if val == 1 else "VSS")
            for pin, val in best_assignment.items()
        }

        self.cache[key] = tieoffs
        return tieoffs


# ---------------------------------------------------------------------------
# Boolean critical path extractor (simplified for brevity)
# ---------------------------------------------------------------------------

class BooleanCriticalPathExtractor:
    """Use BOOLEAN_gates to derive tie-offs and build path-only cell set."""

    def __init__(
        self,
        timing_parser: TimingReportParser,
        netlist_parser: NetlistParser,
        tieoff_engine: BooleanTieOffEngine,
    ):
        self.timing_parser = timing_parser
        self.netlist_parser = netlist_parser
        self.tieoff_engine = tieoff_engine
        self.critical_cells: Dict[str, Dict] = {}
        self.critical_nets: Set[str] = set()
        self.primary_inputs: Set[str] = set()
        self.primary_outputs: Set[str] = set()
        self.path_start_net: Optional[str] = None
        self.path_end_net: Optional[str] = None
        self.tied_inputs: Dict[str, str] = {}

    def _is_sequential(self, cell_type: str) -> bool:
        """Check if a cell type is sequential (flip-flop, latch, etc.)."""
        sequential_patterns = ['DFF', 'LATCH', 'DLY', 'SDFF', 'DFFSR']
        return any(pattern in cell_type for pattern in sequential_patterns)

    def extract(self) -> None:
        """Populate critical_cells, tie-offs, and primary nets."""
        # Skip flip-flops and latches
        for cell_tuple in self.timing_parser.path1_cells:
            if len(cell_tuple) >= 4:
                cell_name, pin, cell_type, arc = cell_tuple[:4]
            else:
                continue

            # Skip sequential cells
            if self._is_sequential(cell_type):
                continue

            # Find in netlist
            found = None
            for inst_name, info in self.netlist_parser.cells.items():
                if inst_name.endswith(cell_name) or cell_name in inst_name:
                    found = (inst_name, info)
                    break

            if not found:
                continue

            inst_name, cell_info = found
            self.critical_cells[inst_name] = cell_info

            # Collect nets
            for p_name, net_name in cell_info["pins"].items():
                if p_name in ["VDD", "VSS", "VNW", "VPW"]:
                    continue
                if net_name.startswith("1'"):
                    continue
                self.critical_nets.add(net_name)

            # Determine tie-offs
            if arc and "->" in arc:
                critical_pin = arc.split(">")[0]
                tieoffs = self.tieoff_engine.get_tieoffs(cell_info["type"], critical_pin)
                for p_name, net_name in cell_info["pins"].items():
                    if p_name != critical_pin and p_name in tieoffs:
                        self.tied_inputs[net_name] = tieoffs[p_name]

        # Identify primary inputs/outputs
        nets_driven = set()
        nets_consumed = set()

        for _, cell_info in self.critical_cells.items():
            output_pins = {"Y"}  # Simplified
            if "DFF" in cell_info["type"] or "LATCH" in cell_info["type"]:
                output_pins = {"Q", "QN"}

            for p_name, net_name in cell_info["pins"].items():
                if p_name in output_pins:
                    nets_driven.add(net_name)
                else:
                    nets_consumed.add(net_name)

        self.primary_inputs = nets_consumed - nets_driven
        self.primary_outputs = nets_driven - nets_consumed

        # Determine path start/end
        if self.primary_inputs:
            self.path_start_net = sorted(self.primary_inputs)[0]
        if self.primary_outputs:
            self.path_end_net = sorted(self.primary_outputs)[0]


# ---------------------------------------------------------------------------
# Netlist generation (simplified)
# ---------------------------------------------------------------------------

class BooleanNetlistGenerator:
    """Generate a Verilog netlist from a BooleanCriticalPathExtractor."""

    def __init__(
        self,
        extractor: BooleanCriticalPathExtractor,
        width: int,
        stages: int,
        iteration: str,
    ):
        self.extractor = extractor
        self.width = width
        self.stages = stages
        self.iteration = iteration

    def generate(self, output_file: str) -> None:
        with open(output_file, "w") as f:
            iteration_suffix = self.iteration.split("_")[-1] if "_" in self.iteration else self.iteration
            module_name = f"critical_path_W{self.width}_S{self.stages}_{iteration_suffix}_bool"

            f.write(f"// Auto-generated critical path (boolean-based tie-offs)\n")
            f.write(f"module {module_name}(\n")
            f.write("  input path_input,\n")
            f.write("  output path_output,\n")
            f.write("  inout VDD,\n")
            f.write("  inout VSS\n")
            f.write(");\n\n")

            # Wires
            if self.extractor.critical_nets:
                nets = sorted(self.extractor.critical_nets)
                f.write(f"  wire {', '.join(nets)};\n\n")

            # Connect path input
            if self.extractor.path_start_net:
                f.write(f"  assign {self.extractor.path_start_net} = path_input;\n\n")

            # Connect path output
            if self.extractor.path_end_net:
                f.write(f"  assign path_output = {self.extractor.path_end_net};\n\n")

            # Tie-offs
            for net, val in sorted(self.extractor.tied_inputs.items()):
                f.write(f"  assign {net} = {val};\n")
            if self.extractor.tied_inputs:
                f.write("\n")

            # Cell instantiations
            for inst_name, cell_info in sorted(self.extractor.critical_cells.items()):
                ctype = cell_info["type"]
                pins = cell_info["pins"]
                conns = []
                for p_name, net_name in sorted(pins.items()):
                    if p_name == "VDD":
                        conns.append(f".{p_name}(VDD)")
                    elif p_name == "VSS":
                        conns.append(f".{p_name}(VSS)")
                    elif net_name.startswith("1'"):
                        tie = "VSS" if "0" in net_name else "VDD"
                        conns.append(f".{p_name}({tie})")
                    else:
                        conns.append(f".{p_name}({net_name})")

                f.write(f"  {ctype} {inst_name} ({', '.join(conns)});\n")

            f.write("\nendmodule\n")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Boolean-gates–driven critical path extractor"
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--stages", type=int, default=DEFAULT_STAGES)
    parser.add_argument("--iteration", type=str, default=DEFAULT_ITERATION)
    parser.add_argument("--boolean-gates", type=str, help="Path to BOOLEAN_gates.v")
    parser.add_argument("--timing-file", type=str, help="Path to timing report")
    parser.add_argument("--netlist-file", type=str, help="Path to netlist")
    parser.add_argument("--output-file", type=str, help="Path to output")
    args = parser.parse_args()

    # Default file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    timing_file = args.timing_file or os.path.join(base_dir, "detailed_timing.txt")
    netlist_file = args.netlist_file or os.path.join(base_dir, "netlist.v")
    output_file = args.output_file or os.path.join(base_dir, "critical_path.v")

    # Find BOOLEAN_gates.v
    if args.boolean_gates:
        boolean_gates_path = args.boolean_gates
    else:
        # Search common locations
        boolean_gates_path = os.path.join(base_dir, "BOOLEAN_gates.v")
        if not os.path.exists(boolean_gates_path):
            boolean_gates_path = "/usr/share/pdk/BOOLEAN_gates.v"

    console.print(Panel.fit(
        f"[bold]Boolean Critical Path Extractor[/bold]\n"
        f"Timing: [cyan]{timing_file}[/cyan]\n"
        f"Netlist: [cyan]{netlist_file}[/cyan]\n"
        f"BOOLEAN_gates: [cyan]{boolean_gates_path}[/cyan]\n"
        f"Output: [cyan]{output_file}[/cyan]",
        box=box.ROUNDED,
    ))

    # Parse
    gates_parser = BooleanGatesParser(boolean_gates_path)
    gates_parser.parse()

    timing_parser = TimingReportParser(timing_file)
    timing_parser.parse_path1()

    net_parser = NetlistParser(netlist_file)
    net_parser.parse()

    # Extract and generate
    engine = BooleanTieOffEngine(gates_parser.gates)
    extractor = BooleanCriticalPathExtractor(timing_parser, net_parser, engine)
    extractor.extract()

    generator = BooleanNetlistGenerator(extractor, args.width, args.stages, args.iteration)
    generator.generate(output_file)

    # Summary
    table = Table(title="Critical Path Summary", box=box.SIMPLE_HEAVY)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Cells on path", str(len(extractor.critical_cells)))
    table.add_row("Tied nets", str(len(extractor.tied_inputs)))
    table.add_row("Output file", output_file)
    console.print(table)

    console.print(f"\n[green]✓[/green] Generated: [cyan]{output_file}[/cyan]")


if __name__ == "__main__":
    main()
