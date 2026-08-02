import argparse
import os
import numpy as np
import plotly.graph_objects as go

def main():
    parser = argparse.ArgumentParser(description="Generate 3D Scatter Plotly Viewer")
    parser.add_argument("--orig", required=False, help="(Ignored) Original volume kept for compatibility")
    parser.add_argument("--pred", required=True, help="Predicted volume `.npy` file")
    parser.add_argument("--out", required=True, help="Output HTML file path")
    parser.add_argument("--max-points", type=int, default=150000, help="Maximum number of points per class to render")
    args = parser.parse_args()

    print(f"Loading predicted volume {args.pred}...")
    pred_vol = np.load(args.pred)
    
    D, H, W = pred_vol.shape

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Extracting coordinates for Pores (Class 2)...")
    # Coordinates where class == 2
    z_p, y_p, x_p = np.where(pred_vol == 2)
    
    print("Extracting coordinates for Chamber (Class 1)...")
    z_c, y_c, x_c = np.where(pred_vol == 1)

    # Downsample pores to prevent browser crashes
    if len(z_p) > args.max_points:
        print(f"Downsampling Pores from {len(z_p)} to {args.max_points} points for visualization...")
        idx = np.random.choice(len(z_p), args.max_points, replace=False)
        z_p, y_p, x_p = z_p[idx], y_p[idx], x_p[idx]

    # Downsample chamber
    if len(z_c) > args.max_points:
        print(f"Downsampling Chamber from {len(z_c)} to {args.max_points} points for visualization...")
        idx = np.random.choice(len(z_c), args.max_points, replace=False)
        z_c, y_c, x_c = z_c[idx], y_c[idx], x_c[idx]
        
    print("Building Plotly 3D scatter plot...")
    
    fig = go.Figure()

    # Add Chamber Trace
    if len(z_c) > 0:
        fig.add_trace(go.Scatter3d(
            x=x_c, y=y_c, z=z_c,
            mode='markers',
            marker=dict(
                size=2,
                color='rgba(255, 225, 25, 0.2)', # Semi-transparent yellow
            ),
            name='Chamber',
            visible='legendonly'  # Hide by default so it doesn't obstruct the pores
        ))

    # Add Pores Trace
    if len(z_p) > 0:
        fig.add_trace(go.Scatter3d(
            x=x_p, y=y_p, z=z_p,
            mode='markers',
            marker=dict(
                size=3,
                color='rgba(50, 205, 50, 0.9)', # Lime green
            ),
            name='Pores'
        ))

    # Maintain proper 3D aspect ratio (Plotly uses scaled axis ranges)
    fig.update_layout(
        title=f"3D Segmentation Volume: {os.path.basename(args.pred)}",
        scene=dict(
            xaxis=dict(title='X Axis', range=[0, W]),
            yaxis=dict(title='Y Axis', range=[0, H]),
            zaxis=dict(title='Z Axis (Slices)', range=[0, D]),
            aspectmode='data' # Ensures a sphere looks like a sphere
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.8, y=0.9),
        paper_bgcolor='#111',
        font=dict(color='white')
    )

    print(f"Saving 3D HTML viewer to {args.out}. This might take a minute...")
    fig.write_html(args.out)
    print("Done! You can now view the HTML file in your browser.")

if __name__ == "__main__":
    main()
