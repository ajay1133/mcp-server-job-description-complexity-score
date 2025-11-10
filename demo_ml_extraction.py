"""Demo script showing ML-powered technology extraction capabilities."""

from mcp_server.ml_tech_extractor import MLTechExtractor
import json


def main():
    print("=" * 70)
    print("ML-Powered Technology Extraction Demo")
    print("=" * 70)
    print()

    # Initialize ML extractor
    print("[1/4] Initializing ML models...")
    extractor = MLTechExtractor()
    print("✓ Models loaded (using trained models or fallback to baselines)")
    print()

    # Example 1: Job description
    print("[2/4] Extracting from job description...")
    job_desc = """
    Senior Full-Stack Engineer
    
    We're looking for an experienced engineer with 5+ years of React and 
    Node.js development. Strong knowledge of PostgreSQL and Docker required.
    Kubernetes experience is a plus.
    """
    
    result = extractor.extract_technologies(job_desc, is_resume=False)
    
    print("Technologies detected:")
    for tech_name, info in result["technologies"].items():
        print(f"  • {tech_name.upper()}")
        print(f"    - Difficulty: {info['difficulty']:.1f}/10 (ML-predicted)")
        print(f"    - Category: {info['category']}")
        
        if "experience_mentioned_in_prompt" in info:
            print(f"    - Experience required: {info['experience_mentioned_in_prompt']} years")
        
        if info["alternatives"]:
            alts = list(info["alternatives"].keys())[:2]
            print(f"    - Alternatives: {', '.join(alts)}")
        print()

    # Example 2: Resume
    print("[3/4] Extracting from resume...")
    resume = """
    Software Engineer
    
    • 3 years experience building web applications with React and TypeScript
    • Backend development with Node.js (2 years) and Express
    • Database design with PostgreSQL
    • Containerization using Docker
    """
    
    result = extractor.extract_technologies(resume, is_resume=True)
    
    print("Resume technologies:")
    for tech_name, info in result["technologies"].items():
        exp_str = ""
        if "experience_accounted_for_in_resume" in info:
            exp_str = f" ({info['experience_accounted_for_in_resume']}y experience)"
        print(f"  • {tech_name}{exp_str} - Difficulty: {info['difficulty']:.1f}/10")
    print()

    # Example 3: Resume + Job matching
    print("[4/4] Matching resume to job requirements...")
    
    job_requirement = "Looking for 5+ years React experience"
    
    result = extractor.extract_technologies(
        text=resume,
        is_resume=True,
        prompt_override=job_requirement
    )
    
    print("Requirement Analysis:")
    for tech_name, info in result["technologies"].items():
        if tech_name == "react":
            prompt_exp = info.get("experience_mentioned_in_prompt", 0)
            resume_exp = info.get("experience_accounted_for_in_resume", 0)
            
            print(f"\n  React:")
            print(f"    Required: {prompt_exp} years")
            print(f"    Candidate has: {resume_exp} years")
            
            if resume_exp >= prompt_exp:
                print(f"    ✓ Meets requirement")
            else:
                gap = prompt_exp - resume_exp
                print(f"    ✗ Gap: {gap} years")
                
                # Suggest alternatives
                if info["alternatives"]:
                    print(f"    💡 Candidate may have transferable skills in:")
                    for alt, alt_info in list(info["alternatives"].items())[:2]:
                        print(f"       - {alt} (similar difficulty: {alt_info['difficulty']:.1f}/10)")

    print()
    print("=" * 70)
    print("Demo complete!")
    print()
    print("Next steps:")
    print("  1. Train models with more data: python train_system_design_models.py")
    print("  2. Integrate with your application")
    print("  3. See mcp_server/ml_models/README.md for details")
    print("=" * 70)


if __name__ == "__main__":
    main()
