# Visualizing the agent's refinement in ParaView

The `export_vtk.py` script writes one `.vti` (VTK ImageData) file per simulation
step into `paraview_output/`. Each file carries three point-data arrays:

| Array     | Meaning                                              |
|-----------|------------------------------------------------------|
| `field`   | the scalar simulation field at this step             |
| `error`   | per-cell interpolation error proxy (gradient mag.)   |
| `refined` | 1 where the agent refined this cell, 0 otherwise     |

## Opening the time series

1. Launch ParaView.
2. **File → Open**, navigate to `paraview_output/`. ParaView automatically
   groups files named `amr_000.vti`, `amr_001.vti`, ... into a single time
   series — select the grouped entry `amr_..vti` and click **OK**.
3. Click **Apply** in the Properties panel.

## Suggested visualization

**See the field evolve**
- In the toolbar coloring dropdown, choose `field`.
- Press the play button to step through time and watch the blob diffuse.

**See where the agent refined**
- Select the dataset, then **Filters → Common → Threshold**.
- Set *Scalars* to `refined`, lower threshold `0.5`, upper `1.0`, click
  **Apply**. Only the refined cells remain — this is the agent's policy made
  visible.
- Colour the threshold output a solid colour and overlay it on a semi-transparent
  copy of the full `field` for context.

**Compare against the error field**
- Add a second view (**Split Horizontal**), colour by `error`, and play both
  views in sync. A well-trained agent's `refined` cells should track the bright
  (high-error) regions of the `error` field.

## Comparing learned vs random

Generate a random-policy rollout into a separate folder and load both:

```bash
python src/export_vtk.py --random --outdir paraview_output_random --steps 30
```

Open both time series and place them side by side. The learned policy should
keep refinement focused on the moving high-gradient front, while the random
policy scatters refinement across the domain.

## Making a screenshot or animation for your portfolio

- **File → Save Screenshot** for a still.
- **File → Save Animation** to export an `.avi`/`.png` sequence stepping through
  the time series — a short clip of the refinement tracking the diffusing front
  makes a strong portfolio visual.
