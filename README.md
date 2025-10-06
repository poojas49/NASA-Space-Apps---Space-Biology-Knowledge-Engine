
***NasaSpace-Apps- ThinkLink***

This project visualizes research papers and their relationships using an **interactive D3.js-based knowledge graph**, making it easier to explore relevant academic literature given a user’s hypothesis or query.

## ✨ Features

- **Chat-driven search**: User inputs a hypothesis or keywords.
- **Backend-powered enrichment**: Query is sent to a FastAPI backend that returns a list of enriched papers and their relationship scores.
- **Interactive Knowledge Graph**:
  - Central node for hypothesis.
  - Surrounding nodes for each paper, sized/scaled by relevance.
  - Links show relationship strength between papers.
  - Click a paper to view title, abstract, authors, and extracted figures.
- **Comparison mode**: Compare two papers side-by-side.
- **Avatar leveling system**: Tracks user exploration behavior and levels up based on number of papers explored.
- **Audio & haptics**: Ambient feedback for interaction (mute toggle supported).

## 🧠 Project Structure

### 🧩 Frontend (React + D3.js)

**Key Components:**

- `KnowledgeGraph.tsx`: Main graph visualization using D3 force simulation.
- `PaperPanel.tsx`: Right-side drawer showing metadata, authors, figures, and comparison actions.
- `ComparisonView.tsx`: Side-by-side comparison of two selected papers.
- `ResearchAvatar.tsx`: Avatar gamification element showing user progress.
- `audioManager.ts`, `haptics.ts`: Manage sound effects and haptic feedback.

**Graph Construction:**

- Hypothesis is pinned to the center of the graph.
- Nodes represent papers with `paper_id`, `title`, `relevance_score`, etc.
- Links come from the `relationships` array and are filtered for validity.
- On node click: Paper metadata and figures are shown in the side panel.
- Hovering shows a tooltip with author details.

### ⚙️ Backend (FastAPI)

- Receives POST request with `{ query: string }`
- Returns:
  - `papers`: list of papers with `paper_id`, `title`, `abstract`, `authors`, `relevance_score`, `figures[]`, etc.
  - `relationships`: list of paper ID pairs with `relationship_strength` (float)
- Optional: Figures are pre-extracted from PDFs and served from `/src/data/extracted_images/{paper_id}`

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/research-knowledge-graph.git
cd research-knowledge-graph
2. Install frontend dependencies
bash
Always show details

Copy code
npm install
# or
yarn install
3. Run the frontend dev server
bash
Always show details

Copy code
npm run dev
Make sure your FastAPI backend is running on http://localhost:8000.

4. Backend setup
Install FastAPI, Uvicorn, etc.



🔧 Technologies Used
Tech	Purpose
React + TypeScript	Frontend UI
D3.js	Graph visualization
FastAPI	Backend paper enrichment API
Supabase	Session & user tracking
Framer Motion	UI animations
Tailwind CSS	Styling
Vite	Dev server and build tool****
