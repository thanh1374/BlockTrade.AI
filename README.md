my_project/Tracing/Desktop/BlockTrace AI/README.md
# BlockTrace AI

Ethereum wallet analysis and risk assessment tool built with Streamlit.

## Features
- Wallet Overview
- Transaction History
- Risk Assessment
- Portfolio Trend Analysis

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install dev dependencies: `pip install black pylint pre-commit`
4. Set up pre-commit hooks: `pre-commit install`
5. Run the app: `streamlit run streamlit_app.py`

## Development
### Code Formatting
- Format code: `black .`
- Check formatting: `black --check .`

### Linting
- Run linter: `pylint app/ streamlit_app.py`
- Fix common issues: `black .`

### Pre-commit Hooks
Pre-commit hooks will automatically format and lint your code before commits.