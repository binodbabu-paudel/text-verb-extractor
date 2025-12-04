"""
Utility functions for the project
"""

import json
import csv
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import matplotlib.pyplot as plt


def save_results(results: Dict[str, Any], output_dir: str = "output"):
    """
    Save analysis results to multiple formats
    
    Args:
        results: Analysis results dictionary
        output_dir: Output directory path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    json_path = os.path.join(output_dir, f"analysis_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save verb list as TXT
    txt_path = os.path.join(output_dir, f"verbs_{timestamp}.txt")
    with open(txt_path, 'w') as f:
        f.write("VERBS EXTRACTED FROM TEXT\n")
        f.write("=" * 40 + "\n\n")
        for verb, count in sorted(results.get('verb_counts', {}).items(), 
                                key=lambda x: x[1], reverse=True):
            f.write(f"{verb.upper():20} : {count} occurrence{'s' if count > 1 else ''}\n")
    
    # Save as CSV
    if 'sentence_analysis' in results:
        csv_path = os.path.join(output_dir, f"sentences_{timestamp}.csv")
        df = pd.DataFrame(results['sentence_analysis'])
        df.to_csv(csv_path, index=False)
    
    return {
        "json": json_path,
        "txt": txt_path,
        "csv": csv_path if 'sentence_analysis' in results else None
    }


def create_visualization(results: Dict[str, Any], save_path: str = None):
    """
    Create visualization of analysis results
    
    Args:
        results: Analysis results dictionary
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Top verbs bar chart
    if 'most_common_verbs' in results:
        ax1 = axes[0, 0]
        verbs, counts = zip(*results['most_common_verbs'])
        colors = plt.cm.Set3(range(len(verbs)))
        ax1.barh(verbs, counts, color=colors)
        ax1.set_xlabel('Frequency')
        ax1.set_title('Top Verbs by Frequency')
        ax1.invert_yaxis()
    
    # 2. Sentence statistics
    ax2 = axes[0, 1]
    labels = ['With Verbs', 'Without Verbs']
    sizes = [
        results.get('sentences_with_verbs', 0),
        results.get('sentences_without_verbs', 0)
    ]
    colors = ['lightgreen', 'lightcoral']
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    ax2.set_title('Sentences with vs without Verbs')
    
    # 3. Verb distribution
    if 'verb_counts' in results:
        ax3 = axes[1, 0]
        verb_counts = results['verb_counts']
        if len(verb_counts) <= 10:
            ax3.pie(verb_counts.values(), labels=verb_counts.keys(), autopct='%1.1f%%')
        else:
            # Show top 10 + others
            sorted_counts = sorted(verb_counts.items(), key=lambda x: x[1], reverse=True)
            top_10 = sorted_counts[:10]
            other_count = sum(count for _, count in sorted_counts[10:])
            
            sizes = [count for _, count in top_10] + [other_count]
            labels = [verb for verb, _ in top_10] + ['Others']
            ax3.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax3.set_title('Verb Distribution')
    
    # 4. Text statistics
    ax4 = axes[1, 1]
    stats = {
        'Sentences': results.get('total_sentences', 0),
        'Words': results.get('total_words', 0),
        'Unique Verbs': results.get('total_unique_verbs', 0),
        'Verb Instances': results.get('total_verb_instances', 0)
    }
    ax4.bar(stats.keys(), stats.values(), color=['blue', 'green', 'red', 'purple'])
    ax4.set_ylabel('Count')
    ax4.set_title('Text Statistics')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def print_summary(results: Dict[str, Any]):
    """
    Print a summary of the analysis results
    
    Args:
        results: Analysis results dictionary
    """
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n📊 Text Statistics:")
    print(f"   • Sentences: {results.get('total_sentences', 0)}")
    print(f"   • Words: {results.get('total_words', 0)}")
    print(f"   • Sentences with verbs: {results.get('sentences_with_verbs', 0)}")
    print(f"   • Sentences without verbs: {results.get('sentences_without_verbs', 0)}")
    
    print(f"\n🔍 Verb Analysis:")
    print(f"   • Unique verbs: {results.get('total_unique_verbs', 0)}")
    print(f"   • Total verb instances: {results.get('total_verb_instances', 0)}")
    print(f"   • Average verbs per sentence: "
          f"{results.get('total_verb_instances', 0) / max(results.get('total_sentences', 1), 1):.2f}")
    
    if 'most_common_verbs' in results:
        print(f"\n📈 Top 5 Most Common Verbs:")
        for verb, count in results['most_common_verbs'][:5]:
            print(f"   • {verb.capitalize()}: {count} times")
