import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
import json
from matplotlib.ticker import FuncFormatter

def format_ticks(x, pos):
    """Format large numbers for better readability (e.g., 1000 -> 1K, 1000000 -> 1M)"""
    if x >= 1e6:
        return f'{x*1e-6:.0f}M'
    elif x >= 1e3:
        return f'{x*1e-3:.0f}K'
    else:
        return f'{x:.0f}'

def read_config_json(tsv_path):
    """Read the config.json file in the same directory as the TSV file"""
    try:
        # Get the directory containing the TSV file
        directory = os.path.dirname(tsv_path)
        config_path = os.path.join(directory, 'config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        else:
            print(f"Config file not found at {config_path}")
            return None
    except Exception as e:
        print(f"Error reading config file: {e}")
        return None

def read_and_plot_success_curves(tsv_path):
    """Read TSV file and plot individual success rate curves"""
    try:
        # Read TSV file
        df = pd.read_csv(tsv_path, sep='\t')
        
        if 'total_env_steps' not in df.columns:
            print(f"Column 'total_env_steps' not found in file {tsv_path}")
            return None
        
        # Extract method name (from file path)
        method_name = os.path.basename(os.path.dirname(tsv_path))
        
        # Read config.json file
        config = read_config_json(tsv_path)
        buffer_type = config.get('buffer_type', 'unknown') if config else 'unknown'
        cl_method = config.get('cl_method', 'none') if config else 'none'
        if cl_method is None:
            cl_method = 'none'
        
        # Include config info in the method name for titles
        method_info = f"{method_name} (buffer: {buffer_type}, method: {cl_method})"
        
        # Find success columns, separating stochastic and deterministic
        stochastic_columns = [col for col in df.columns if 'success' in col and 'stochastic' in col and 'train/success' not in col]
        deterministic_columns = [col for col in df.columns if 'success' in col and 'deterministic' in col and 'train/success' not in col]
        
        if not stochastic_columns and not deterministic_columns:
            print(f"No 'success' columns found in file {tsv_path}")
            return None
        
        # Create output directory
        output_dir = 'success_curves'
        os.makedirs(output_dir, exist_ok=True)
        
        # Process stochastic success columns
        stochastic_figs = []
        for col in stochastic_columns:
            # Extract task name from column name
            task_match = re.search(r'/([\w\-]+)/success', col)
            if task_match:
                task_name = task_match.group(1)
            else:
                task_name = col.replace('test/stochastic/', '').replace('/success', '')
            
            # Create a new figure for this success metric
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the success curve
            if 'average_success' in col:
                line_style = '-'
                line_width = 3
                marker = 'o'
                marker_size = 5
            else:
                line_style = '--'
                line_width = 2
                marker = None
                marker_size = 0
                
            ax.plot(df['total_env_steps'], df[col], 
                    label=f"{task_name}", 
                    linewidth=line_width, 
                    linestyle=line_style,
                    marker=marker,
                    markersize=marker_size)
            
            # Set up chart
            ax.set_xlabel('Training Steps', fontsize=12)
            ax.set_ylabel('Success Rate', fontsize=12)
            
            # Special title for average success
            if 'average_success' in col:
                ax.set_title(f'Average Stochastic Success Rate\n{method_info}', fontsize=14)
                output_filename = f'stochastic_avg_{method_name}.png'
            else:
                ax.set_title(f'Task "{task_name}" Stochastic Success Rate\n{method_info}', fontsize=14)
                output_filename = f'stochastic_{task_name}_{method_name}.png'
                
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Format x-axis ticks
            formatter = FuncFormatter(format_ticks)
            ax.xaxis.set_major_formatter(formatter)
            
            # Set y-axis range (success rates are typically between 0 and 1)
            ax.set_ylim([-0.05, 1.05])
            
            # Add legend
            ax.legend(fontsize=10)
            
            plt.tight_layout()
            
            # Save the figure
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=300)
            
            print(f"Stochastic success curve for {task_name} saved to: {output_path}")
            stochastic_figs.append(fig)
        
        # Process deterministic success columns
        deterministic_figs = []
        for col in deterministic_columns:
            # Extract task name from column name
            task_match = re.search(r'/([\w\-]+)/success', col)
            if task_match:
                task_name = task_match.group(1)
            else:
                task_name = col.replace('test/deterministic/', '').replace('/success', '')
            
            # Create a new figure for this success metric
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the success curve
            if 'average_success' in col:
                line_style = '-'
                line_width = 3
                marker = 'o'
                marker_size = 5
            else:
                line_style = '--'
                line_width = 2
                marker = None
                marker_size = 0
                
            ax.plot(df['total_env_steps'], df[col], 
                    label=f"{task_name}", 
                    linewidth=line_width, 
                    linestyle=line_style,
                    marker=marker,
                    markersize=marker_size)
            
            # Set up chart
            ax.set_xlabel('Training Steps', fontsize=12)
            ax.set_ylabel('Success Rate', fontsize=12)
            
            # Special title for average success
            if 'average_success' in col:
                ax.set_title(f'Average Deterministic Success Rate\n{method_info}', fontsize=14)
                output_filename = f'deterministic_avg_{method_name}.png'
            else:
                ax.set_title(f'Task "{task_name}" Deterministic Success Rate\n{method_info}', fontsize=14)
                output_filename = f'deterministic_{task_name}_{method_name}.png'
                
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Format x-axis ticks
            formatter = FuncFormatter(format_ticks)
            ax.xaxis.set_major_formatter(formatter)
            
            # Set y-axis range (success rates are typically between 0 and 1)
            ax.set_ylim([-0.05, 1.05])
            
            # Add legend
            ax.legend(fontsize=10)
            
            plt.tight_layout()
            
            # Save the figure
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=300)
            
            print(f"Deterministic success curve for {task_name} saved to: {output_path}")
            deterministic_figs.append(fig)
        
        return stochastic_figs + deterministic_figs
    
    except Exception as e:
        print(f"Error processing file {tsv_path}: {e}")
        return None

def compare_methods_plot(policy_type="stochastic", task_name="average_success"):
    """
    Create a comparison plot of multiple methods similar to the example figure
    
    Args:
        policy_type: Either "stochastic" or "deterministic"
        task_name: Task to compare or "average_success" for overall comparison
    """
    # Define colors for different methods
    method_colors = {
        'agem': '#70C1B3',       # A-GEM (green-blue)
        'ewc': '#7F7F7F',        # EWC (gray)
        'grow': '#B39DDB',       # Grow (light purple)
        'cod': '#3D9970',        # CoD (teal)
        'lwf': '#CCCC00',        # LWF (yellow)
        'mas': '#673AB7',        # MAS (purple)
        'packnet': '#388E3C',    # PackNet (green)
        'prune': '#FF69B4',      # Prune (pink)
        'rwalk': '#CD853F',      # RWalk (brown)
        'vcl': '#808080',        # VCL (gray)
        'finetuning': '#FF0000', # Finetuning (red)
        'none': '#0000FF'        # None (blue)
    }
    
    # Method name mapping for display
    method_display_names = {
        'agem': 'A-GEM', 
        'ewc': 'EWC', 
        'grow': 'Grow', 
        'cod': 'CoD',
        'lwf': 'LWF', 
        'mas': 'MAS', 
        'packnet': 'PackNet', 
        'prune': 'Prune',
        'rwalk': 'RWalk', 
        'vcl': 'VCL',
        'finetuning': 'Finetuning',
        'none': 'None'
    }
    
    # Create figure with clean, clear style
    plt.figure(figsize=(14, 8))
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        # Fallback to default style with grid
        plt.style.use('default')
        plt.grid(True, linestyle='-', alpha=0.2)
    
    # Dictionary to store data for each method
    methods_data = {}
    
    # Find all progress.tsv files for different methods
    for path in glob.glob('cl/cl_0/progress.tsv'):
        # Extract method name from directory
        dir_name = os.path.basename(os.path.dirname(path))
        
        try:
            # Read the TSV file
            df = pd.read_csv(path, sep='\t')
            
            # Check if required columns exist
            if 'total_env_steps' not in df.columns:
                continue
            
            # Read config to get actual cl_method
            config = read_config_json(path)
            if config:
                cl_method = config.get('cl_method', 'none')
                if cl_method is None:
                    cl_method = 'none'
            else:
                cl_method = 'unknown'
            
            # Determine which success column to use
            if task_name == "average_success":
                column_pattern = f'test/{policy_type}/average_success'
            else:
                column_pattern = f'test/{policy_type}/{task_name}/success'
            
            # Find the matching column
            matching_cols = [col for col in df.columns if column_pattern in col]
            
            if matching_cols:
                # Get the data and store it
                methods_data[dir_name] = {
                    'cl_method': cl_method,
                    'steps': df['total_env_steps'].values,
                    'success': df[matching_cols[0]].values
                }
                print(f"Added data for {dir_name} (method: {cl_method})")
        
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    if not methods_data:
        print("No data found for the specified policy type and task.")
        return
    
    # Plot each method
    for method_name, data in methods_data.items():
        cl_method = data['cl_method']
        
        # Get color for the method
        color = method_colors.get(cl_method, None)
        
        # Get display name for the method
        display_name = method_display_names.get(cl_method, cl_method)
        
        # If no specific color for this method, use a random one
        if color is None:
            color = np.random.rand(3,)
        
        # Plot the data with thinner lines and no markers for cleaner appearance
        plt.plot(data['steps'], data['success'], 
                 label=display_name, 
                 color=color,
                 linewidth=1.5)
    
    # Set title based on task
    if task_name == "average_success":
        title_task = "Performance across methods on the OCW10 sequence"
    else:
        title_task = f"Performance across methods on task '{task_name}'"
    
    # Customize plot appearance
    plt.title(title_task, fontsize=14)
    plt.xlabel('# of Training Steps (5e4 for Each Task)', fontsize=12)
    plt.ylabel('Average Success Rate', fontsize=12)
    
    # Set y-axis range appropriate for success rates
    plt.ylim([0, max(0.8, max([max(data['success']) for data in methods_data.values()]) + 0.1)])
    
    # Add legend with multiple columns if many methods
    if len(methods_data) > 5:
        plt.legend(fontsize=10, loc='upper left', ncol=3)
    else:
        plt.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    # Create output directory
    output_dir = 'success_curves'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    output_filename = f'comparison_{policy_type}_{task_name}.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    
    print(f"Comparison plot saved to: {output_path}")
    return plt.gcf()

def main():
    """Main function to run the script with various options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process and plot TSV data files.')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'compare'],
                        help='Mode: single (plot single method) or compare (compare methods)')
    parser.add_argument('--method', type=str, default='cl_0',
                        help='Method directory name (for single mode)')
    parser.add_argument('--policy', type=str, default='stochastic',
                        choices=['stochastic', 'deterministic'],
                        help='Policy type to analyze')
    parser.add_argument('--task', type=str, default='average_success',
                        help='Task name to analyze or "average_success" for overall performance')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Find the progress file for specified method
        path = f'saved_logs/cl/{args.method}/progress.tsv'
        if os.path.exists(path):
            print(f"Processing file: {path}")
            figs = read_and_plot_success_curves(path)
            
            # Close all figures to avoid memory issues
            if figs:
                for fig in figs:
                    plt.close(fig)
            
            print(f"All charts for {args.method} generated successfully")
        else:
            print(f"No progress.tsv file found for {args.method}. Please check the file path.")
    
    elif args.mode == 'compare':
        # Generate comparison plot
        fig = compare_methods_plot(args.policy, args.task)
        plt.close(fig)
        print("Comparison plot generated successfully")

if __name__ == "__main__":
    # If no arguments provided, run default comparison
    import sys
    if len(sys.argv) == 1:
        print("Generating default comparison plot...")
        fig = compare_methods_plot("stochastic", "average_success")
        plt.close(fig)
    else:
        main()
