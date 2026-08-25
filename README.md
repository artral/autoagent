# How to run when you want to migrate a file
python3 spec_agent.py --migrate userService.js --target "MS SQL stored procedures"
# How to run when you want to migrate multiple  files
python3 spec_agent.py --migrate userService.js orderService.js --target "MS SQL stored procedures"
# Just to generate a file from a spec from scratch
python3 spec_agent.py spec.md
```
Since build_migration_spec() produces a plain str, everything after it in the pipeline treats it identically to a hand-written spec — generate_plan still produces a schema-validated JSON plan (now with a data_flow field that maps each JS function to its target stored procedure, per the instruction we added), you still get the human-review pause, and generate_code → validate_code will now write real .sql files and run them through the sqlfluff --dialect tsql parser we set up earlier. Nothing new to wire up on that end.

One thing worth testing on your side since I can't make a real LLM call here: the actual quality of the migration mapping — whether the model correctly translates parameterized queries, ? placeholders, and JS control flow into equivalent T-SQL constructs — is a model-behavior question, not a plumbing question, so worth running it against a real file and checking the plan's data_flow/assumptions fields before trusting the generated procedures.
```
# env usage
# Specify multiple files separated by a comma or space
MIGRATE_FILES="userService.js, orderService.js"

# Specify the target database system
TARGET_DB="MS SQL stored procedures"

import os
from argparse import ArgumentParser
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

parser = ArgumentParser()
# Use the env variable as a fallback default if the flag is missing
parser.add_argument('--migrate', nargs='+', default=os.getenv('MIGRATE_FILES', '').replace(',', ' ').split())
parser.add_argument('--target', default=os.getenv('TARGET_DB'))

args = parser.parse_args()
