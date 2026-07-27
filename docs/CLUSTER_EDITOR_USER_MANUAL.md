# Cluster Editor User Manual

This document explains the daily workflow for `src/post_processing/cluster_editor_vue.py`.

## Start the Editor

```bash
python src/post_processing/cluster_editor_vue.py data/cluster_state/ --port 5005
```

Open `http://localhost:5005` in your browser.

## Core Workflow

1. Open a volume from the top dropdown.
2. Use `Select` mode to paint pores that should be edited.
3. Apply `Merge` or `Split` operation.
4. Repeat until cluster labels are correct.
5. Click `Save`.

## Controls

### Camera Navigation
- Left drag: rotate
- Right drag: pan
- Mouse wheel: zoom

### Paint Modes
- `Select`: add pores to current selection
- `Eraser`: remove pores from current selection
- `Shift + mouse wheel` (or `Shift + two-finger scroll`): adjust brush radius
- In Select/Eraser, hold `Shift` while dragging left mouse button to rotate camera

## Chamber Panel
- `Show/Hide`: toggle chamber visibility
- Double-click a chamber row: solo that chamber (double-click again to restore all)
- Row `Merge` button: move selected pores directly into that chamber
- `Show Chamber Context`: display shell/body context points
- `Opacity`: adjust context visibility

## Edit Operations

### Merge
- If pores are selected: merges selected pores' chambers together
- If nothing is selected: merges all currently visible chambers

### Split
- Requires selected pores
- Creates a new chamber from the selection

### Undo
- Toolbar `Undo` button
- Or `Ctrl+Z`

## Save Output

`Save` writes outputs into `corrected_volumes` next to your state folder:
- `<volume>_corrected.npy`
- `<volume>_corrected_mapping.npz`

## Keyboard Shortcuts
- `Ctrl+Z`: undo last action
- `Esc`: close Help modal (if open), otherwise switch to Navigate mode

## Tips
- Use visibility controls before merge to avoid unintended merges.
- Keep brush size smaller for boundary corrections.
- Save after each major edit pass.
