# Setup Time Allocation for Non-Coding Technologies

## Problem
Previously, technologies like CDN, Cassandra, Redis, monitoring, and auth services were assigned:
- `loc_overhead: 0` (correct, as they don't require custom code)
- `time_overhead_hours: 0` (incorrect, as they require setup/configuration time)

This underestimated the total project time.

## Solution
Implemented difficulty-based setup time allocation for non-coding technologies:

### Formula
```
If LOC overhead > 0:
    time = (LOC / 1000) * 0.77 hours    // Coding technologies
Else if difficulty rating exists:
    time = 0.3 * (1.5 ^ difficulty)     // Non-coding technologies (setup/config)
Else:
    time = 0                            // Unknown/skipped technologies
```

### Time by Difficulty Level
| Difficulty | Setup Time | Category |
|------------|------------|----------|
| 1 (Easy) | 27 minutes | Simple config files |
| 2 | 40 minutes | Basic setup |
| 3 | 1.0 hour | Moderate config |
| 4 | 1.5 hours | Standard integration |
| 5 | 2.3 hours | Complex setup |
| 6 | 3.4 hours | Advanced config |
| 7 (Hard) | 5.1 hours | Complex integration |
| 8 | 7.7 hours | Very complex setup |
| 9 | 11.5 hours | Expert-level config |
| 10 (Expert) | 17.3 hours | Extremely complex |

## Example Results

### Before (CDN with difficulty=4):
```json
{
  "technology": "cdn",
  "loc_overhead": 0,
  "time_overhead_hours": 0.0,
  "time_overhead_readable": "0 minutes"
}
```

### After (CDN with difficulty=4):
```json
{
  "technology": "cdn",
  "loc_overhead": 0,
  "time_overhead_hours": 1.52,
  "time_overhead_readable": "1.5 hours"
}
```

## Technologies Affected

**Common non-coding technologies now have realistic setup time:**
- **Auth/OAuth** (difficulty 7): ~5.1 hours
- **DevOps/CI/CD** (difficulty 7): ~5.1 hours
- **Monitoring** (difficulty 5): ~2.3 hours
- **Video Processing** (difficulty 7): ~5.1 hours
- **Streaming** (difficulty 7): ~5.1 hours
- **CDN** (difficulty 4): ~1.5 hours
- **Serverless** (difficulty 5): ~2.3 hours
- **Service Mesh** (difficulty 8): ~7.7 hours

## Impact

**Before:** "build a video streaming platform with CDN"
- Total non-coding setup time: 0 hours ❌

**After:** "build a video streaming platform with CDN"
- Total non-coding setup time: ~30+ hours ✅
- More realistic project estimates
- Better reflects actual DevOps, auth, monitoring, and CDN setup effort

## Technical Details

The setup time calculation is in `_analyze_technology_criticality()`:

```python
# Get difficulty rating from technology_difficulty.json
tech_difficulty = difficulty_map.get(tech, 0)

if loc_overhead > 0:
    # Coding technology: use LOC-based calculation
    time_overhead_hours = (loc_overhead / 1000.0) * 0.77
else:
    # Non-coding technology: use difficulty-based calculation
    if tech_difficulty > 0:
        time_overhead_hours = 0.3 * (1.5 ** tech_difficulty)
    else:
        time_overhead_hours = 0.0
```

## Benefits
1. ✅ More accurate total time estimates
2. ✅ Reflects real-world setup complexity
3. ✅ Difficulty-weighted (harder setups take exponentially more time)
4. ✅ Coding technologies still use LOC-based calculation
5. ✅ Unknown technologies default to 0 (safe fallback)
