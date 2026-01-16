# Prompt: Model Terumo “Workhorse” Guidewires (Procedural Shape, SOFA/BeamAdapter)

You are helping me build **physically plausible, simulation-ready** models of **Terumo “Workhorse” guidewires** (multiple variants) for a SOFA BeamAdapter / stEVE-based simulator. I want **accurate parameters** (preferably from datasheets, IFUs, publications, or direct measurements) and **clear unit handling**.

## 1) Goal

Create parameter sets for several Terumo “Workhorse” wire variants that differ in:
- **Tip shape** (e.g., straight, J-tip, angled/shapeable, helical if applicable)
- **Tip stiffness vs shaft stiffness** (distal flexible tip vs proximal/shaft)
- **Diameter and length variants** (e.g., 0.014", 0.018", 0.035" families—confirm which)
- **Coating/friction behavior** (e.g., hydrophilic vs PTFE vs bare metal) if the simulator supports it

The target representation is a **procedural guidewire** with separate **shaft** vs **tip/extremity** parameters.

## 2) Simulator parameter mapping (what the code expects)

The procedural wire is defined using a “procedural rest shape” with the following fields (shaft vs tip/extremity):

### A) Geometry (floats)
- `length` (total length)
- `straight_length` (shaft length; the remainder is the tip region)
- `spire_diameter` (tip curvature/loop diameter; for a planar J-tip: `spire_diameter = 2 * tip_radius`)
- `spire_height` (helix pitch; for planar tips usually `0`)
- `radius` (shaft outer radius)
- `radius_extremity` (tip outer radius)
- `inner_radius` (shaft inner radius, often `0`)
- `inner_radius_extremity` (tip inner radius, often `0`)

### B) Material (floats; shaft vs tip)
- `poisson_ratio` (0–0.5)
- `young_modulus` (shaft)
- `young_modulus_extremity` (tip)
- `mass_density` (shaft)
- `mass_density_extremity` (tip)

### C) Discretization / simulation (ints + tuples)
- `num_edges` (visual discretization along total length)
- `key_points` (tuple of cumulative arc-length positions; typically `(0.0, straight_length, length)`)
- `num_edges_collis` (tuple of collision edges per segment; length = `len(key_points)-1`)
- `density_of_beams` (tuple of FEM beams per segment; length = `len(key_points)-1`)
- `velocity_limit` at device level: `(translation_mm_s, rotation_rad_s)`
- `color` at device level: `(r, g, b)`

If a requested tip shape cannot be represented with this procedural parameterization, explicitly tell me and propose an alternative (e.g., a piecewise arc/mesh-based rest shape), but still provide the best procedural approximation.

## 3) What I need from you (your deliverables)

For each Terumo “Workhorse” wire variant you model, produce:

1. **A parameter table** with:
   - parameter name
   - value
   - units
   - source (URL / document name / measurement method)
   - notes/assumptions
2. **Derived values and formulas** you used (e.g., converting tip radius/angle into `straight_length` and `spire_diameter`).
3. A **simulation-ready** output in two formats:
   - `tool_definition.json` `spec` block (all numeric parameters)
   - a minimal `tool.py` class skeleton using those values (shaft vs tip clearly separated)
4. **Clarifying questions** for anything you cannot infer without guessing.

Important: **Do not invent numbers.** If a value is missing, label it as `unknown` and propose how to obtain it (datasheet lookup or a measurement protocol), or provide a range clearly marked as an estimate.

## 4) The values I need (types + unit expectations)

### Required (per variant)

**Identification**
- `name` (string)
- `description` (string)

**Geometry** (all floats)
- Total length (e.g., `length_mm`)
- Outer diameter at shaft and tip (e.g., `outer_diameter_mm`, or separate `straight_outer_diameter_mm` / `tip_outer_diameter_mm`)
- Inner diameter (usually `0`, unless a lumen exists)
- Tip shape parameters (choose one set that matches the tip type):
  - For J-tip / circular arc: `tip_radius_mm` + `tip_angle_deg` (or `tip_length_mm`)
  - For straight tip: `tip_length_mm` (and set curvature to 0)
  - For helix/spire: `spire_diameter_mm` + `spire_height_mm` (+ number of turns or tip length)

**Material** (all floats)
- `poisson_ratio` (0.0–0.5)
- Tip Young’s modulus vs shaft Young’s modulus (two values)
- Tip mass density vs shaft mass density (two values)

**Simulation/discretization** (floats + derived ints)
- `visu_edges_per_mm` (float) → used to compute `num_edges`
- `collis_edges_per_mm_tip` (float) and `collis_edges_per_mm_shaft` (float)
- `beams_per_mm_tip` (float) and `beams_per_mm_shaft` (float)
- `velocity_limit_mm_s__rad_s` (tuple `[translation_mm_s, rotation_rad_s]`)

### Optional but useful (if available)
- Distal flexible zone length(s) beyond a single “tip” (e.g., 30–50 mm) and how to approximate it with a 2-region model
- Coating type and an estimated friction coefficient for contact (if the simulator exposes `mu`)
- Tip load / tip stiffness metrics used clinically (and how to map them to `young_modulus_extremity`)

## 5) Units: confirm and be explicit

My simulator uses **millimeters** for geometry (`mm`). For mechanical values, I need you to:
- State the unit system you assume.
- Output values in **both** SI and “mm-consistent” form if relevant.

Use these conversions when needed:
- `E_MPa = E_Pa / 1e6` (since `1 MPa = 1 N/mm²`)
- `rho_kg_mm3 = rho_kg_m3 / 1e9` (since `1 m³ = 1e9 mm³`)

If you’re unsure which unit convention my scene uses internally, ask me to confirm before finalizing numbers.

## 6) Output format (please follow exactly)

### A) Variant summary (human-readable)
- Bullet list of modeled variants
- For each: short clinical description + what differs (tip shape, stiffness, coating)

### B) Parameter table (per variant)
Provide a Markdown table with columns:
`Parameter | Value | Units | Source | Notes`

### C) JSON (`tool_definition.json`) snippet (per variant)
Return a JSON object with:
- `name`, `description`, `type: "procedural"`, and a `spec` object containing all numeric fields (including tip + shaft + discretization + velocity limits).

### D) Python (`tool.py`) snippet (per variant)
Return a minimal dataclass skeleton that clearly maps the `spec` parameters into a procedural device definition (shaft vs tip/extremity separated).

## 7) Questions you should ask me first (if not specified)

1. Which exact “Terumo Workhorse” products/part numbers am I targeting (and which diameter family)?
2. Do I want one **generic** Workhorse model or multiple variants (e.g., straight vs J vs shapeable)?
3. What is my simulator’s **unit convention** for Young’s modulus and density (SI vs mm-consistent)?
4. Do I have datasheets/IFUs, or should you only use publicly accessible sources?

