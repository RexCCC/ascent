"""Generate activation profiles for nerve fibers under varying contact current conditions.

The copyrights of this software are owned by Duke University.
Please refer to the LICENSE and README.md files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent.

This script implements the project goal outlined in Project_planning.txt:
- Fix the current on contact 1 at 400 μA
- Vary the current on contact 2 in increments (0%, 20%, 40%, 60%, 80%, 100% of 400 μA)
- Visualize activation profiles (red for activated fibers, grey for non-activated)

Run this from repository root.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pandas as pd

# Add the repository root to the path if necessary
try:
    from src.core.plotter import activation_profile
    from src.core.query import Query
    from src.utils.enums import Object  # Import Object enum
except ImportError:
    print("Make sure to run this script from the repository root")
    sys.exit(1)

# Define the conditions from the project plan
CONDITIONS = [
    {"p_percent": 0, "sim_array": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "I_total_uA": 400},
    {"p_percent": 20, "sim_array": [0.8333, 0.1667, 0.0, 0.0, 0.0, 0.0], "I_total_uA": 480},
    {"p_percent": 40, "sim_array": [0.7143, 0.2857, 0.0, 0.0, 0.0, 0.0], "I_total_uA": 560},
    {"p_percent": 60, "sim_array": [0.625, 0.375, 0.0, 0.0, 0.0, 0.0], "I_total_uA": 640},
    {"p_percent": 80, "sim_array": [0.5556, 0.4444, 0.0, 0.0, 0.0, 0.0], "I_total_uA": 720},
    {"p_percent": 100, "sim_array": [0.5, 0.5, 0.0, 0.0, 0.0, 0.0], "I_total_uA": 800}
]

def print_activation_table(condition, plotter):
    """
    Print a detailed table showing activation status for all fibers in the current condition
    
    Parameters:
    condition (dict): Current condition info
    plotter (_ActivationProfilePlotter): Plotter object with activation data
    
    Returns:
    pd.DataFrame: The activation data DataFrame
    """
    df = plotter.activation_df
    
    # Add condition information
    p_percent = condition["p_percent"]
    contact1_uA = 400  # Fixed
    contact2_uA = (p_percent / 100) * 400
    total_current_uA = condition["I_total_uA"]
    total_current_mA = total_current_uA / 1000.0
    
    print(f"\nDetailed Fiber Activation Table for Condition: p={p_percent}%")
    print(f"Contact 1: {contact1_uA} μA, Contact 2: {contact2_uA} μA, Total: {total_current_uA} μA ({total_current_mA} mA)")
    print("-" * 100)
    
    # Print detailed info for first 10 fibers
    print("Sample of fiber activation data (first 10 fibers):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print(df.head(10).to_string())
    print()
    
    # Print activation statistics per inner fascicle
    if 'inner' in df.columns:
        inner_stats = df.groupby('inner').agg({
            'is_activated': ['count', 'sum', lambda x: 100 * sum(x) / len(x)]
        })
        inner_stats.columns = ['Total Fibers', 'Activated Fibers', 'Activation %']
        print("\nActivation Statistics by Inner Fascicle:")
        print(inner_stats.to_string())
        print()
        
    # Activation threshold distribution
    print("\nActivation Threshold Distribution:")
    thresh_bins = [0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43]
    thresh_counts = []
    for i in range(len(thresh_bins)-1):
        count = ((df['threshold_mA'] >= thresh_bins[i]) & 
                 (df['threshold_mA'] < thresh_bins[i+1])).sum()
        thresh_counts.append(count)
    
    for i in range(len(thresh_bins)-1):
        print(f"  {thresh_bins[i]:.2f} - {thresh_bins[i+1]:.2f} mA: {thresh_counts[i]} fibers")
    
    print("-" * 100)
    
    # Return the DataFrame for potential further use
    return df, inner_stats

def debug_sample_fiber_data(q):
    """
    Debug function to examine the sample and fiber data
    
    Parameters:
    q (Query): The query object with loaded data
    """
    print("\n===== DEBUG SAMPLE DATA =====")
    
    # Get sample from query using the Object enum
    sample = q.get_object(Object.SAMPLE, [2])
    if not sample:
        print("No sample found!")
        return
        
    print(f"Sample type: {type(sample)}")
    print(f"Sample attributes: {dir(sample)[:20]}")
    
    # Check if sample has slides
    if hasattr(sample, 'slides') and sample.slides:
        print(f"Sample has {len(sample.slides)} slides")
        slide = sample.slides[0]
        print(f"Slide type: {type(slide)}")
        print(f"Slide attributes: {dir(slide)[:20]}")
        
        # Check fascicles
        if hasattr(slide, 'fascicles') and slide.fascicles:
            print(f"Slide has {len(slide.fascicles)} fascicles")
            fascicle = slide.fascicles[0]
            print(f"Fascicle type: {type(fascicle)}")
            
            # Check inners
            if hasattr(fascicle, 'inners') and fascicle.inners:
                print(f"Fascicle has {len(fascicle.inners)} inners")
                inner = fascicle.inners[0]
                print(f"Inner type: {type(inner)}")
                print(f"Inner area: {inner.area() if hasattr(inner, 'area') else 'N/A'}")
    
    # Check if sample has fiber_set
    print("\n----- Fiber Set Data -----")
    if hasattr(sample, 'fiber_set') and sample.fiber_set:
        fiber_set = sample.fiber_set
        print(f"Fiber set type: {type(fiber_set)}")
        print(f"Fiber set attributes: {dir(fiber_set)[:20]}")
        
        # Check fibers
        if hasattr(fiber_set, 'fibers') and fiber_set.fibers:
            print(f"Fiber set has {len(fiber_set.fibers)} fibers")
            print(f"First fiber type: {type(fiber_set.fibers[0])}")
            
            # Show sample fiber
            if isinstance(fiber_set.fibers[0], dict):
                fiber_keys = list(fiber_set.fibers[0].keys())
                print(f"First fiber keys: {fiber_keys}")
                if 'fiber' in fiber_keys:
                    print(f"First fiber['fiber'] type: {type(fiber_set.fibers[0]['fiber'])}")
                    print(f"First fiber['fiber'] sample: {fiber_set.fibers[0]['fiber'][:2]}")
            elif isinstance(fiber_set.fibers[0], list):
                print(f"First fiber length: {len(fiber_set.fibers[0])}")
                print(f"First fiber sample: {fiber_set.fibers[0][:2]}")
    else:
        print("Sample has no fiber_set attribute or it's None")
    
    # Check sim data
    print("\n----- Simulation Data -----")
    sim = q.get_object(Object.SIMULATION, [2, 0, 8])
    if sim:
        print(f"Sim type: {type(sim)}")
        print(f"Sim attributes: {dir(sim)[:20]}")
        
        # Check for fiber information in simulation
        if hasattr(sim, 'fibers'):
            print(f"Sim has fibers attribute: {type(sim.fibers)}")
            if hasattr(sim.fibers, 'fibers'):
                print(f"Sim.fibers has {len(sim.fibers.fibers)} fibers")
    else:
        print("No simulation object found")
        
    print("===== END DEBUG =====\n")

def main():
    """
    Main function to generate activation profiles using the new plotter function
    """
    # Initialize and run Query to get threshold data
    q = Query({
        'partial_matches': True,
        'include_downstream': True,
        'indices': {'sample': [2], 'model': [0], 'sim': [8]},
        'model_filters': [lambda m: isinstance(m.get('cuff'), dict)]  # Only use dict-type cuffs
    }).run()
    
    # Debug sample and fiber data
    debug_sample_fiber_data(q)
    
    # Get threshold data
    threshold_data = q.threshold_data(ignore_missing=True)
    
    # Get sample and simulation objects to pass to the plotter
    # This is important for correct fiber position plotting
    sample_index = 2
    model_index = 0
    sim_index = 8
    
    sample_object = q.get_object(Object.SAMPLE, [sample_index])
    sim_object = q.get_object(Object.SIMULATION, [sample_index, model_index, sim_index])
    
    # Create output directory
    save_directory = os.path.join('out', 'activation_profiles')
    os.makedirs(save_directory, exist_ok=True)
    
    # Create a directory for CSV files
    csv_directory = os.path.join(save_directory, 'csv')
    os.makedirs(csv_directory, exist_ok=True)
    
    # Store all the activation data for potential additional analysis
    all_activation_data = {}
    
    # Generate visualizations for each condition
    image_paths = []
    
    for condition in CONDITIONS:
        p_percent = condition["p_percent"]
        sim_array = condition["sim_array"]
        I_total = condition["I_total_uA"]
        
        # Calculate contact currents based on percentages
        contact1_uA = 400  # Fixed current on contact 1
        contact2_uA = (p_percent / 100) * 400  # Variable current on contact 2
        
        print(f"\n{'='*40}")
        print(f"Generating visualization for condition: {p_percent}%")
        print(f"  Contact 1: {contact1_uA} μA")
        print(f"  Contact 2: {contact2_uA} μA")
        print(f"  Total Current: {I_total} μA")
        print(f"{'='*40}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get nsim=0 data for each condition
        nsim_data = threshold_data[threshold_data['nsim'] == 0]
        
        # Use the activation_profile plotter
        # Pass sample_object and sim_object explicitly to ensure proper fiber positioning
        plotter = activation_profile(
            data=nsim_data,
            ax=ax,
            currents={'contact1_uA': contact1_uA, 'contact2_uA': contact2_uA},
            cutoff_current=I_total,
            cuff_array=sim_array,
            contact_colors=["red", "blue"],  # Red for contact 1, blue for contact 2
            sample_object=sample_object,  # Pass the sample object explicitly
            sim_object=sim_object,        # Pass the simulation object explicitly
            _return_plotter=True          # Special flag to return the plotter object too
        )
        
        # Print detailed activation table for this condition and get the DataFrame
        activation_df, inner_stats = print_activation_table(condition, plotter)
        
        # Save the activation data to CSV
        csv_file = os.path.join(csv_directory, f'activation_data_p{p_percent}.csv')
        activation_df.to_csv(csv_file, index=False)
        print(f"  Activation data saved to: {csv_file}")
        
        # Save inner statistics to CSV if available
        if inner_stats is not None:
            inner_csv = os.path.join(csv_directory, f'inner_stats_p{p_percent}.csv')
            inner_stats.to_csv(inner_csv)
            print(f"  Inner statistics saved to: {inner_csv}")
        
        # Store the data for potential further analysis
        all_activation_data[p_percent] = {
            'df': activation_df,
            'inner_stats': inner_stats,
            'condition': condition
        }
        
        # Save the figure
        output_path = os.path.join(save_directory, f'activation_profile_p{p_percent}.png')
        fig.savefig(output_path, dpi=400, bbox_inches='tight')
        image_paths.append(output_path)
        
        print(f"  Visualization saved to: {output_path}")
        plt.close(fig)
    
    # Save all activation data to a single Excel file with multiple sheets
    excel_file = os.path.join(csv_directory, 'all_activation_data.xlsx')
    with pd.ExcelWriter(excel_file) as writer:
        for p_percent, data in all_activation_data.items():
            # Save the fiber data
            sheet_name = f'Fibers_p{p_percent}'
            data['df'].to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Save the inner statistics
            if data['inner_stats'] is not None:
                stats_sheet = f'Stats_p{p_percent}'
                data['inner_stats'].to_excel(writer, sheet_name=stats_sheet)
    
    print(f"\nComplete activation data saved to Excel file: {excel_file}")
    
    # Create animation from the saved images
    try:
        images = []
        for path in image_paths:
            images.append(imageio.imread(path))
        
        # Save as GIF with 1 frame per second
        output_gif = os.path.join(save_directory, 'activation_profiles_animation.gif')
        imageio.mimsave(output_gif, images, fps=1)
        print(f"Created animation: {output_gif}")
    except Exception as e:
        print(f"Error creating animation: {e}")
    
    print("\nAnalysis complete. You can now interact with the activation data in the CSV and Excel files.")

if __name__ == "__main__":
    main() 