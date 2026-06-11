from pathlib import Path
import sys
Root_Folder = Path(__file__).resolve().parent.parent
if str(Root_Folder) not in sys.path: sys.path.insert(0, str(Root_Folder))