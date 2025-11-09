#!/usr/bin/env python3
"""Show setup time calculation formula."""

print("Setup Time Formula: 0.3 * (1.5 ^ difficulty)\n")
print(f"{'Difficulty':<12} {'Hours':<15} {'Human Readable'}")
print("-" * 50)

difficulties = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for d in difficulties:
    hours = 0.3 * (1.5 ** d)
    if hours < 1:
        readable = f"{hours * 60:.0f} minutes"
    elif hours < 24:
        readable = f"{hours:.1f} hours"
    elif hours < 168:
        readable = f"{hours/24:.1f} days"
    else:
        readable = f"{hours/168:.1f} weeks"
    
    print(f"{d:<12} {hours:<15.2f} {readable}")

print("\n✅ Exponential growth ensures harder setups take significantly more time")
