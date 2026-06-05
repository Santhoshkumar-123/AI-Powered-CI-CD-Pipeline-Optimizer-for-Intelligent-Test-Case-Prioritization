"""
Generate Architecture Diagrams for IEEE Paper
Run: python generate_diagrams.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style for IEEE paper
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

def create_system_architecture():
    """Create overall system architecture diagram"""
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    color_data = '#E3F2FD'
    color_feature = '#FFF3E0'
    color_ai = '#E8F5E9'
    color_engine = '#F3E5F5'
    color_viz = '#FCE4EC'
    
    # Layer 1: Data Collection
    ax.add_patch(FancyBboxPatch((0.5, 10), 9, 1.5, boxstyle="round,pad=0.1", 
                                facecolor=color_data, edgecolor='black', linewidth=2))
    ax.text(5, 11, 'Data Collection Layer', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2, 10.5, 'Code\nRepository', ha='center', va='center', fontsize=9)
    ax.text(5, 10.5, 'Test Execution\nData', ha='center', va='center', fontsize=9)
    ax.text(8, 10.5, 'Historical\nMetrics', ha='center', va='center', fontsize=9)
    
    # Layer 2: Feature Extraction
    ax.add_patch(FancyBboxPatch((0.5, 7.5), 9, 2, boxstyle="round,pad=0.1", 
                                facecolor=color_feature, edgecolor='black', linewidth=2))
    ax.text(5, 9.2, 'Feature Extraction Layer', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Statistical Features
    ax.add_patch(FancyBboxPatch((1, 7.8), 3.5, 1.2, boxstyle="round,pad=0.05", 
                                facecolor='white', edgecolor='blue', linewidth=1))
    ax.text(2.75, 8.7, 'Statistical Features', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(2.75, 8.3, '• Code Churn\n• Execution Time\n• Failure History', 
            ha='center', va='center', fontsize=7)
    
    # Semantic Features
    ax.add_patch(FancyBboxPatch((5.5, 7.8), 3.5, 1.2, boxstyle="round,pad=0.05", 
                                facecolor='white', edgecolor='blue', linewidth=1))
    ax.text(7.25, 8.7, 'Semantic Features', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(7.25, 8.3, '• Semantic Vectors\n• Drift Phase\n• Code Embeddings', 
            ha='center', va='center', fontsize=7)
    
    # Layer 3: AI Model
    ax.add_patch(FancyBboxPatch((0.5, 5), 9, 2, boxstyle="round,pad=0.1", 
                                facecolor=color_ai, edgecolor='black', linewidth=2))
    ax.text(5, 6.7, 'AI Model Layer', ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.add_patch(FancyBboxPatch((2, 5.3), 6, 1.2, boxstyle="round,pad=0.05", 
                                facecolor='white', edgecolor='green', linewidth=2))
    ax.text(5, 6.2, 'Semanti-Q Agent', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, 5.8, 'Deep Q-Network\n(128 → 64 → Q-values)', 
            ha='center', va='center', fontsize=8)
    
    # Layer 4: Prioritization Engine
    ax.add_patch(FancyBboxPatch((0.5, 3), 9, 1.5, boxstyle="round,pad=0.1", 
                                facecolor=color_engine, edgecolor='black', linewidth=2))
    ax.text(5, 4.2, 'Prioritization Engine', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(3.5, 3.5, 'Test Ranking', ha='center', va='center', fontsize=9)
    ax.text(6.5, 3.5, 'Execution Order', ha='center', va='center', fontsize=9)
    
    # Layer 5: Visualization Dashboard
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 9, 2, boxstyle="round,pad=0.1", 
                                facecolor=color_viz, edgecolor='black', linewidth=2))
    ax.text(5, 2.2, 'Visualization Dashboard', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2, 1.2, 'Metrics\nDisplay', ha='center', va='center', fontsize=8)
    ax.text(4, 1.2, 'Risk\nHeatmap', ha='center', va='center', fontsize=8)
    ax.text(6, 1.2, 'AI\nInsights', ha='center', va='center', fontsize=8)
    ax.text(8, 1.2, 'Analytics', ha='center', va='center', fontsize=8)
    
    # Add arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    ax.annotate('', xy=(5, 9.5), xytext=(5, 10), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 7), xytext=(5, 7.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 4.5), xytext=(5, 5), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 2.5), xytext=(5, 3), arrowprops=arrow_props)
    
    plt.title('Semanti-Q System Architecture', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: system_architecture.png")
    plt.close()

def create_neural_network_architecture():
    """Create Semanti-Q neural network architecture"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Input layer
    ax.text(1, 7, 'Input Layer', ha='center', fontsize=11, fontweight='bold')
    
    # Statistical features
    for i in range(5):
        circle = plt.Circle((1, 5.5 - i*0.5), 0.15, color='lightblue', ec='black')
        ax.add_patch(circle)
    ax.text(1, 2.5, 'Statistical\nFeatures (n)', ha='center', fontsize=9)
    
    # Semantic features
    for i in range(3):
        circle = plt.Circle((1, 1.5 - i*0.3), 0.15, color='lightgreen', ec='black')
        ax.add_patch(circle)
    ax.text(1, 0.3, 'Semantic\nVector (10)', ha='center', fontsize=9)
    
    # Hidden Layer 1
    ax.text(4, 7, 'Hidden Layer 1', ha='center', fontsize=11, fontweight='bold')
    for i in range(8):
        circle = plt.Circle((4, 5.5 - i*0.7), 0.15, color='#FFE082', ec='black')
        ax.add_patch(circle)
    ax.text(4, 0.3, '128 neurons\nReLU', ha='center', fontsize=9)
    
    # Hidden Layer 2
    ax.text(6.5, 7, 'Hidden Layer 2', ha='center', fontsize=11, fontweight='bold')
    for i in range(6):
        circle = plt.Circle((6.5, 5 - i*0.8), 0.15, color='#FFAB91', ec='black')
        ax.add_patch(circle)
    ax.text(6.5, 0.3, '64 neurons\nReLU', ha='center', fontsize=9)
    
    # Output Layer
    ax.text(9, 7, 'Output Layer', ha='center', fontsize=11, fontweight='bold')
    for i in range(3):
        circle = plt.Circle((9, 4 - i*0.8), 0.15, color='#EF9A9A', ec='black')
        ax.add_patch(circle)
    ax.text(9, 0.3, 'Q-Values', ha='center', fontsize=9)
    
    # Draw connections (sample)
    for i in range(5):
        for j in range(8):
            ax.plot([1.15, 3.85], [5.5 - i*0.5, 5.5 - j*0.7], 
                   'gray', alpha=0.2, linewidth=0.5)
    
    for i in range(8):
        for j in range(6):
            ax.plot([4.15, 6.35], [5.5 - i*0.7, 5 - j*0.8], 
                   'gray', alpha=0.2, linewidth=0.5)
    
    for i in range(6):
        for j in range(3):
            ax.plot([6.65, 8.85], [5 - i*0.8, 4 - j*0.8], 
                   'gray', alpha=0.2, linewidth=0.5)
    
    plt.title('Semanti-Q Agent Neural Network Architecture', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('neural_network_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: neural_network_architecture.png")
    plt.close()

def create_training_pipeline():
    """Create training pipeline flowchart"""
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define box positions
    boxes = [
        (5, 13, 'Raw Test Data\n600 tests, 100 cycles', '#E3F2FD'),
        (5, 11.5, 'Data Preprocessing', '#FFF3E0'),
        (5, 10, 'Feature Extraction', '#FFF3E0'),
        (2.5, 8.5, 'Training Set\nCycles 0-19', '#E8F5E9'),
        (7.5, 8.5, 'Test Set\nCycles 20-99', '#E8F5E9'),
        (5, 7, 'Semanti-Q Agent\nTraining', '#C8E6C9'),
        (5, 5.5, 'Q-Learning\n5 Epochs', '#A5D6A7'),
        (5, 4, 'Model Checkpoint', '#81C784'),
        (5, 2.5, 'Online Simulation', '#F3E5F5'),
        (5, 1, 'Performance Evaluation', '#FCE4EC'),
    ]
    
    for x, y, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=color, edgecolor='black', linewidth=1.5))
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    ax.annotate('', xy=(5, 12.6), xytext=(5, 13.4), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 11.1), xytext=(5, 11.9), arrowprops=arrow_props)
    ax.annotate('', xy=(2.5, 9.6), xytext=(4.5, 10.4), arrowprops=arrow_props)
    ax.annotate('', xy=(7.5, 9.6), xytext=(5.5, 10.4), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 7.4), xytext=(2.5, 8.1), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 6.6), xytext=(5, 7.4), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 5.1), xytext=(5, 5.9), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 3.6), xytext=(5, 4.4), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 2.9), xytext=(5, 3.6), arrowprops=arrow_props)
    ax.annotate('', xy=(5.5, 2.5), xytext=(7.5, 8.1), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 1.4), xytext=(5, 2.1), arrowprops=arrow_props)
    
    # Add metrics boxes
    metrics = [
        (2, 0.2, 'APFD'),
        (5, 0.2, 'Time Savings'),
        (8, 0.2, 'Stability'),
    ]
    for x, y, text in metrics:
        ax.add_patch(FancyBboxPatch((x-0.8, y-0.15), 1.6, 0.3, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='#FFECB3', edgecolor='orange', linewidth=1))
        ax.text(x, y, text, ha='center', va='center', fontsize=8)
    
    ax.annotate('', xy=(2, 0.5), xytext=(4, 0.6), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 0.5), xytext=(5, 0.6), arrowprops=arrow_props)
    ax.annotate('', xy=(8, 0.5), xytext=(6, 0.6), arrowprops=arrow_props)
    
    plt.title('Semanti-Q Training Pipeline', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('training_pipeline.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: training_pipeline.png")
    plt.close()

def create_results_chart():
    """Create APFD comparison chart"""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    methods = ['Random', 'History-Based', 'Coverage-Based', 'Semanti-Q\n(Proposed)']
    apfd_scores = [0.52, 0.68, 0.71, 0.78]
    colors = ['#BDBDBD', '#90CAF9', '#81C784', '#4CAF50']
    
    bars = ax.bar(methods, apfd_scores, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, score in zip(bars, apfd_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('APFD Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Prioritization Method', fontsize=12, fontweight='bold')
    ax.set_title('APFD Comparison: Semanti-Q vs Baseline Methods', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('apfd_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: apfd_comparison.png")
    plt.close()

if __name__ == "__main__":
    print("Generating architecture diagrams for IEEE paper...")
    print("-" * 50)
    
    create_system_architecture()
    create_neural_network_architecture()
    create_training_pipeline()
    create_results_chart()
    
    print("-" * 50)
    print("✓ All diagrams generated successfully!")
    print("\nGenerated files:")
    print("  1. system_architecture.png")
    print("  2. neural_network_architecture.png")
    print("  3. training_pipeline.png")
    print("  4. apfd_comparison.png")
    print("\nThese diagrams are ready for your IEEE paper!")
