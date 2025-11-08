#!/usr/bin/env python3
"""
Recalculate human vs AI ratio from existing data with adjustment:
- Add prompt overhead to AI time: extra_time = total_lines / (100 * human_lines_per_sec)

Note: We do NOT halve human time because breaks apply to both manual and AI-assisted development.
"""

import json
from datetime import datetime

def recalculate_with_adjustments(input_file: str, output_log: str):
    """Recalculate metrics with adjustments applied."""
    
    # Load existing data
    with open(input_file, 'r') as f:
        old_results = json.load(f)
    
    print(f"Loaded {len(old_results)} repos from {input_file}")
    print("\nRecalculating with adjustment:")
    print("  - AI time + prompt overhead = total_lines / (100 × human_lines_per_sec)")
    print("  - Human time NOT adjusted (breaks apply to estimates too)")
    print()
    
    new_results = []
    
    for i, old in enumerate(old_results, 1):
        print(f"[{i}/{len(old_results)}] {old['repo_name']}")
        
        repo_name = old['repo_name']
        total_lines = old['total_lines']
        num_contributors = old['num_contributors']
        
        # Use original human time (no halving)
        human_time_seconds = old['human_time_taken_seconds']
        
        # Calculate human lines per second
        human_lines_per_second = total_lines / (num_contributors * human_time_seconds)
        
        # Get original AI base time
        ai_time_base_seconds = old['ai_time_taken_seconds']
        
        # ADJUSTMENT: Add prompt overhead
        extra_user_prompt_to_ai_time = total_lines / (100 * human_lines_per_second)
        
        # Total AI time
        ai_time_seconds = ai_time_base_seconds + extra_user_prompt_to_ai_time
        
        # Recalculate AI lines per second
        ai_lines_per_second = total_lines / ai_time_seconds
        
        # Recalculate time ratio
        time_ratio = human_lines_per_second / ai_lines_per_second
        
        print(f"  Human: {human_time_seconds/86400:.1f} days (original, not halved)")
        print(f"  Old AI: {ai_time_base_seconds/3600:.1f}h → New AI: {ai_time_seconds/3600:.1f}h (+ {extra_user_prompt_to_ai_time/3600:.2f}h prompt)")
        print(f"  Old ratio: {old['time_ratio']:.8f} → New ratio: {time_ratio:.8f}")
        print(f"  Human is {time_ratio*100:.6f}% of AI speed (AI is {1/time_ratio:.2f}x faster)")
        print()
        
        new_results.append({
            'repo_name': repo_name,
            'repo_creation_time': old['repo_creation_time'],
            'last_commit_time': old['last_commit_time'],
            'total_lines': total_lines,
            'num_contributors': num_contributors,
            'human_time_taken_seconds': human_time_seconds,
            'human_time_taken_days': human_time_seconds / 86400,
            'ai_time_base_seconds': ai_time_base_seconds,
            'ai_time_base_hours': ai_time_base_seconds / 3600,
            'prompt_overhead_seconds': extra_user_prompt_to_ai_time,
            'prompt_overhead_hours': extra_user_prompt_to_ai_time / 3600,
            'ai_time_taken_seconds': ai_time_seconds,
            'ai_time_taken_hours': ai_time_seconds / 3600,
            'human_lines_per_sec': human_lines_per_second,
            'ai_lines_per_sec': ai_lines_per_second,
            'time_ratio': time_ratio,
            'old_time_ratio': old['time_ratio']
        })
    
    # Calculate average ratio
    average_time_ratio = sum(r['time_ratio'] for r in new_results) / len(new_results)
    old_average = sum(r['old_time_ratio'] for r in new_results) / len(new_results)
    
    # Calculate statistics
    time_ratios = [r['time_ratio'] for r in new_results]
    time_ratios_sorted = sorted(time_ratios)
    median_ratio = time_ratios_sorted[len(time_ratios_sorted)//2]
    min_ratio = min(time_ratios)
    max_ratio = max(time_ratios)
    
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Old average ratio: {old_average:.8f} (human was {old_average*100:.6f}% of AI, AI was {1/old_average:.2f}x faster)")
    print(f"New average ratio: {average_time_ratio:.8f} (human is {average_time_ratio*100:.6f}% of AI, AI is {1/average_time_ratio:.2f}x faster)")
    print(f"\nChange: Ratio increased by {(average_time_ratio/old_average - 1)*100:.2f}%")
    print(f"        AI multiplier decreased from {1/old_average:.2f}x to {1/average_time_ratio:.2f}x")
    print("=" * 100)
    print()
    
    # Write detailed log
    print(f"Writing results to {output_log}...")
    
    with open(output_log, 'w', encoding='utf-8') as f:
        f.write("=" * 160 + "\n")
        f.write("GitHub Repository Analysis: Human vs AI Coding Speed (ADJUSTED)\n")
        f.write("=" * 160 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().isoformat()}\n")
        f.write(f"Total repos analyzed: {len(new_results)}\n\n")
        
        f.write("ADJUSTMENT APPLIED:\n")
        f.write("  - AI time includes prompt overhead: extra_time = total_lines / (100 × human_lines_per_sec)\n")
        f.write("    This accounts for time user spends giving prompts to AI\n")
        f.write("  - Human time NOT adjusted (breaks apply to both manual and AI-assisted estimates)\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write(f"  OLD average time ratio (human/AI): {old_average:.8f}\n")
        f.write(f"  NEW average time ratio (human/AI): {average_time_ratio:.8f}\n")
        f.write(f"  This means: Human coding speed is {average_time_ratio*100:.6f}% of AI speed\n")
        f.write(f"  Or equivalently: AI is {1/average_time_ratio:.2f}x faster than human\n\n")
        
        f.write(f"  Change from old calculation:\n")
        f.write(f"    - Ratio increased by {(average_time_ratio/old_average - 1)*100:.2f}%\n")
        f.write(f"    - AI multiplier decreased from {1/old_average:.2f}x to {1/average_time_ratio:.2f}x\n")
        f.write(f"    - This is more realistic because it accounts for time spent giving prompts to AI\n\n")
        
        f.write("STATISTICS:\n")
        f.write(f"  Median ratio: {median_ratio:.8f}\n")
        f.write(f"  Min ratio: {min_ratio:.8f} (fastest human relative to AI)\n")
        f.write(f"  Max ratio: {max_ratio:.8f} (slowest human relative to AI)\n\n")
        
        f.write("-" * 160 + "\n")
        f.write(f"{'Repo Name':<45} {'Lines':>10} {'Contrib':>7} "
                f"{'Human Days':>12} {'AI Base':>10} {'Prompt OH':>11} {'AI Total':>10} "
                f"{'Old Ratio':>12} {'New Ratio':>12} {'Change %':>10}\n")
        f.write("-" * 160 + "\n")
        
        for r in new_results:
            change_pct = (r['time_ratio'] / r['old_time_ratio'] - 1) * 100
            f.write(
                f"{r['repo_name']:<45} "
                f"{r['total_lines']:>10,} "
                f"{r['num_contributors']:>7} "
                f"{r['human_time_taken_days']:>12.1f} "
                f"{r['ai_time_base_hours']:>10.1f} "
                f"{r['prompt_overhead_hours']:>11.2f} "
                f"{r['ai_time_taken_hours']:>10.1f} "
                f"{r['old_time_ratio']:>12.8f} "
                f"{r['time_ratio']:>12.8f} "
                f"{change_pct:>10.1f}%\n"
            )
        
        f.write("-" * 160 + "\n\n")
        
        f.write("=" * 160 + "\n")
        f.write("FINAL RESULTS (ADJUSTED)\n")
        f.write("=" * 160 + "\n\n")
        
        f.write(f"AVERAGE TIME RATIO: {average_time_ratio:.10f}\n\n")
        
        f.write("Interpretation:\n")
        f.write(f"  - Human coding speed is {average_time_ratio*100:.8f}% of AI coding speed\n")
        f.write(f"  - AI is {1/average_time_ratio:.2f}x faster than human coding\n")
        f.write(f"  - For every 1 line AI writes, human writes {average_time_ratio:.8f} lines in the same time\n")
        f.write(f"  - If human takes 1 day, AI takes {average_time_ratio * 24:.6f} hours\n\n")
        
        f.write("Recommendation for model:\n")
        f.write(f"  Use formula: manual_time = ai_time × {1/average_time_ratio:.2f}\n")
        f.write(f"  Or equivalently: ai_time = manual_time × {average_time_ratio:.10f}\n\n")
        
        f.write("Comparison with old calculation:\n")
        f.write(f"  Old formula: manual_time = ai_time × {1/old_average:.2f}\n")
        f.write(f"  New formula: manual_time = ai_time × {1/average_time_ratio:.2f}\n")
        f.write(f"  Difference: New multiplier is {((1/average_time_ratio)/(1/old_average) - 1)*100:.2f}% {'lower' if average_time_ratio > old_average else 'higher'}\n")
    
    # Also save as JSON
    output_json = output_log.replace('.log', '.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            'analysis_date': datetime.now().isoformat(),
            'adjustments': {
                'human_time_halved': False,
                'ai_prompt_overhead_added': True,
                'prompt_overhead_formula': 'total_lines / (100 × human_lines_per_sec)',
                'note': 'Human time not adjusted - breaks apply to both manual and AI-assisted development'
            },
            'summary': {
                'total_repos': len(new_results),
                'old_average_time_ratio': old_average,
                'new_average_time_ratio': average_time_ratio,
                'median_time_ratio': median_ratio,
                'min_time_ratio': min_ratio,
                'max_time_ratio': max_ratio,
                'old_ai_speed_multiplier': 1 / old_average,
                'new_ai_speed_multiplier': 1 / average_time_ratio,
                'human_as_percent_of_ai': average_time_ratio * 100,
                'recommended_formula': f'manual_time = ai_time × {1/average_time_ratio:.2f}'
            },
            'repos': new_results
        }, f, indent=2)
    
    print(f"✓ Log saved to {output_log}")
    print(f"✓ JSON saved to {output_json}")
    print()
    print("=" * 100)
    print("RECOMMENDED FORMULA FOR MODEL:")
    print(f"  manual_time = ai_time × {1/average_time_ratio:.2f}")
    print("=" * 100)

if __name__ == '__main__':
    recalculate_with_adjustments(
        'human_ai_code_ratio_temp.json',
        'logs/human_ai_code_ratio/human_ai_code_ratio.log'
    )
