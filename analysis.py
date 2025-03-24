import json

CONFERENCE_PAPER_PATH = "/data2/czhenghao/paper-distillation/conference-paper/iclr2025.json"

papers = json.load(open(CONFERENCE_PAPER_PATH))

def filter_iclr2025_papers(papers):
    keywords = {"3d", "manipulation", "robot", "depth", "normal", "point cloud", "splat", "mesh", "nerf", "radiance", "inductive", "embodied", "point-cloud", "robotics", "robotic"}
    filtered_papers = []
    
    for paper in papers:
        if paper["status"] in {"Spotlight", "Oral"}:
            title_keywords = set(paper["title"].lower().split())
            keyword_matches = any(word in paper["keywords"].lower() for word in keywords)
            title_matches = any(word in title_keywords for word in keywords)
            
            if keyword_matches or title_matches:
                filtered_papers.append({
                    "title": paper["title"],
                    "status": paper["status"],
                    "rating": paper["rating_avg"],
                    "keywords": paper["keywords"],
                })
    
    print(f"Filtered {len(filtered_papers)} papers")
    return filtered_papers

filtered_papers = filter_iclr2025_papers(papers)
json.dump(filtered_papers, open('/data2/czhenghao/paper-distillation/conference-paper-output/iclr2025_filtered.json', 'w'), indent=4)