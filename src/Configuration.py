from pathlib import Path
import sys

############################################################################
#########################  CONFIGURABLE OPTIONS ############################
############################################################################

# Number of logical CPU cores used for parallel processing for run Models.
#
# Valid values:
#   CPU_COUNT = 0  -> use all available logical CPU cores.
#   CPU_COUNT >= 4 -> use the specified number of logical CPU cores.
#
# Recommended values: 4, 6, or 8.
#
# Values smaller than 4 are not supported.
# The configuration check script will terminate if an invalid value is
# detected.

CPU_COUNT = 4



############################################################################
############################### CONSTANTS ##################################
############################################################################

Root_Folder = Path(__file__).resolve().parent
if str(Root_Folder) not in sys.path: sys.path.insert(0, str(Root_Folder))

PROJECT_FOLDER = Root_Folder.parent
ROOT_FOLDER = PROJECT_FOLDER / "src"
DATA_FOLDER = PROJECT_FOLDER / "data"
REFERENCE_FOLDER = PROJECT_FOLDER / "reference"
OUTPUT_FOLDER = PROJECT_FOLDER / "Output"
RESULT_FOLDER = PROJECT_FOLDER / "Result"



############################################################################
################################# SETUP ####################################
############################################################################
### Create Folders
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)