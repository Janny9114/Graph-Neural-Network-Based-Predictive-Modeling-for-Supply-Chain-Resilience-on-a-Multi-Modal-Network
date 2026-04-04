# Supply Chain Resilience Project - AI Agent Instructions

## Project Overview
This project analyzes supply chain resilience through synthetic data generation and interactive web visualization. It consists of:
- **Python data synthesis** (`graph_synthesis.py`): Generates multi-tier supply chain networks with risk factors using NetworkX and pandas
- **React web dashboard** (`Supply Chain Resilience Website/`): Visualizes resilience metrics, risk distributions, and network maps

## Architecture & Data Flow
- Synthetic supply chains are modeled as directed graphs with 4 tiers: suppliers → manufacturers → distributors → retailers
- Node features include capacity, cost factors, and region-based risk levels (sourced from `ResilienceIndexRegions.csv`)
- Edge features capture lead times, transport costs, and disruption probabilities
- Generated data exports to CSV (`synthetic_nodes.csv`, `synthetic_edges.csv`) for potential ML training or analysis
- Web app currently uses hardcoded mock data; integration with synthetic CSVs is pending

## Key Workflows
- **Data Generation**: Run `python graph_synthesis.py` in virtual environment to create new synthetic networks
- **Web Development**: `cd "Supply Chain Resilience Website"` → `npm install` → `npm run dev` for local development
- **Build**: `npm run build` for production bundle

## Code Patterns & Conventions
- **Python**: Use NetworkX `DiGraph` with node/edge attributes; pandas DataFrames for feature engineering; numpy for distributions
- **React/TypeScript**: Functional components with hooks; shadcn/ui components (Radix primitives + Tailwind); react-simple-maps for geography
- **Styling**: Tailwind CSS with custom theme in `styles/theme.css`; responsive grid layouts
- **Data Handling**: Region risk multipliers loaded from CSV; risk levels clipped to [0.01, 0.99]; Poisson distribution for connection degrees
- **File Structure**: Components in `src/app/components/`; UI primitives in `ui/` subdirectory

## Integration Points
- Resilience scores from `ResilienceIndexRegions.csv` adjust node risks by country
- Web app components expect location data with coordinates, risk levels, and connection mappings
- Future: Load synthetic CSVs into web app for dynamic scenario analysis

## Common Tasks
- Modify tier scaling in `generate_tier_sizes()` for different network sizes
- Add new risk factors by extending node/edge feature generation functions
- Integrate CSV data loading in React components using `d3.csv()` or fetch APIs
- Update map markers by modifying location arrays in `WorldMapNetwork.tsx`

Reference: `graph_synthesis.py` for data models, `App.tsx` for dashboard structure, `ResilienceChart.tsx` for visualization patterns.