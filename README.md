# Critical Path Extractor

Extracts combinational logic cells from a critical timing path and generates a minimal Verilog netlist for standalone timing analysis.

## Overview

This tool parses:
- **Tempus STA timing reports** — extracts Path 1 (critical path)
- **Boolean gate definitions** — `BOOLEAN_gates.v` with primitive logic
- **Design netlists** — full chip netlist

And generates:
- **Minimal critical path netlist** — combinational cells only (no flip-flops/latches)
- **Intelligent tie-offs** — non-critical inputs tied to VDD/VSS using Boolean evaluation
- **Fanout loading** — includes fanout gates to match STA capacitive loading

## Features

- ✅ Boolean-based tie-off derivation (not heuristic)
- ✅ Automatic flip-flop/latch exclusion
- ✅ Fanout gate inclusion for accurate loading
- ✅ VDD/VSS power rail handling
- ✅ Rich CLI output with progress bars

## Installation

```bash
pip install rich
```

## Usage

### Basic

```bash
python extract_critical_path.py \
  --timing-file detailed_timing.txt \
  --netlist-file netlist.v \
  --boolean-gates BOOLEAN_gates.v \
  --output-file critical_path.v
```

### With defaults

```bash
python extract_critical_path.py \
  --width 64 \
  --stages 3 \
  --iteration 002hs
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--timing-file` | Tempus detailed timing report | `detailed_timing.txt` |
| `--netlist-file` | Verilog netlist | `netlist.v` |
| `--boolean-gates` | Boolean gate definitions | `BOOLEAN_gates.v` |
| `--output-file` | Output netlist | `critical_path.v` |
| `--width` | Design width | 64 |
| `--stages` | Pipeline stages | 3 |
| `--iteration` | Iteration suffix | `002hs` |

## How It Works

1. **Parse BOOLEAN_gates.v** — Builds a map of cell base names to their primitive Boolean logic
2. **Parse timing report** — Extracts Path 1 cells, arcs, and transitions
3. **Parse netlist** — Finds cell instances and their connectivity
4. **Extract critical path** — 
   - Skip sequential cells (DFF, LATCH, etc.)
   - For each combinational cell, identify the critical input pin
   - Use Boolean evaluation to find valid tie-offs for side inputs
5. **Generate netlist** —
   - Declare wires for all nets
   - Connect `path_input` to first combinational net
   - Connect `path_output` from last combinational net
   - Apply tie-offs to non-critical inputs
   - Instantiate all cells with proper power pin connections

## Boolean Tie-Off Engine

For each gate on the critical path:

```python
# Example: OAI21 gate with critical input A0
# Try all combinations of side inputs (A1, B0)
# Find assignments where Y can transition when A0 changes

valid_assignments = []
for assignment in all_combinations:
    y_when_a0_0 = eval_gate(A0=0, **assignment)
    y_when_a0_1 = eval_gate(A0=1, **assignment)
    if y_when_a0_0 != y_when_a0_1:
        valid_assignments.append(assignment)

# Select assignment with most VDD ties (better drive strength)
best = max(valid_assignments, key=count_vdd)
```

## Output Example

```verilog
module critical_path_W64_S3_002hs_bool(
  input path_input,
  output path_output,
  inout VDD,
  inout VSS
);

  wire n_4080, n_4081, sum_23_;
  // ... more wires

  assign path_input_net = path_input;
  assign path_output = sum_23_;

  // Tie-offs derived from Boolean analysis
  assign n_4080 = VDD;
  assign n_4081 = VSS;

  // Critical path cells
  OAI21_X2N_A9PP84TR_C14 FE_RC_834_0 (
    .A0(sum_signal_919),
    .A1(n_4080),
    .B0(n_4081),
    .Y(FE_RN_227_0),
    .VDD(VDD),
    .VSS(VSS)
  );
  // ... more cells

endmodule
```

## Requirements

- Python 3.8+
- `rich` — CLI output and progress bars
- `BOOLEAN_gates.v` — Gate primitive definitions for your PDK
- Tempus STA timing report
- Design netlist

## License

MIT
