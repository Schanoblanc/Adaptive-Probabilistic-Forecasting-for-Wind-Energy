import sys
import multiprocessing
import psutil

GREEN = "\033[42m"
YELLOW = "\033[43m"
RED = "\033[41m"
RESET = "\033[0m"

###### Configurtaion Check
import _bootstrap
import Configuration
print()
print("\033[42m CONFIGURATION IMPORT SUCCESS \033[0m")
print()

print(f"PROJECT_FOLDER   : {Configuration.PROJECT_FOLDER}")
print(f"ROOT_FOLDER      : {Configuration.ROOT_FOLDER}")
print(f"DATA_FOLDER      : {Configuration.DATA_FOLDER}")
print(f"REFERENCE_FOLDER : {Configuration.REFERENCE_FOLDER}")
print(f"OUTPUT_FOLDER    : {Configuration.OUTPUT_FOLDER}")
print(f"RESULT_FOLDER    : {Configuration.RESULT_FOLDER}")


###### CPU Count Check
MAX_CPU_COUNT = multiprocessing.cpu_count()
if MAX_CPU_COUNT < 8:
    print()
    print("\033[41m CPU CHECK FAILED \033[0m")
    print(f"Detected CPU cores: {MAX_CPU_COUNT}")
    print("This reproduction package requires at least 8 CPU cores.")
    raise RuntimeError(
        f"CPU requirement not satisfied. Detected CPU cores: {MAX_CPU_COUNT}. Minimum required: 8."
    )

print()
print("\033[42m CPU CHECK PASSED \033[0m")
print(f"Detected CPU cores: {MAX_CPU_COUNT}")
print()


###### CPU Core Configuration Check

CONFIG_CPU_COUNT = Configuration.CPU_COUNT

if CONFIG_CPU_COUNT != 0 and CONFIG_CPU_COUNT < 4:
    print()
    print("\033[41m CPU CORE CONFIGURATION CHECK FAILED \033[0m")
    print(f"Configured CPU_COUNT: {CONFIG_CPU_COUNT}")
    print("CPU_COUNT must be either 0 or an integer greater than or equal to 4.")
    raise RuntimeError(
        f"Invalid CPU_COUNT={CONFIG_CPU_COUNT}. "
        f"CPU_COUNT must be either 0 or >= 4."
    )

if CONFIG_CPU_COUNT > MAX_CPU_COUNT:
    print()
    print("\033[41m CPU CORE CONFIGURATION CHECK FAILED \033[0m")
    print(f"Configured CPU_COUNT: {CONFIG_CPU_COUNT}")
    print(f"Detected logical CPU cores: {MAX_CPU_COUNT}")
    print("Configured CPU_COUNT exceeds the number of available CPU cores.")
    raise RuntimeError(
        f"Invalid CPU_COUNT={CONFIG_CPU_COUNT}. "
        f"Detected available CPU cores: {MAX_CPU_COUNT}."
    )

if CONFIG_CPU_COUNT == 0:
    print()
    print("\033[42m CPU CORE CONFIGURATION CHECK PASSED \033[0m")
    print(f"Configured CPU_COUNT: {CONFIG_CPU_COUNT}")
    print(f"Detected logical CPU cores: {MAX_CPU_COUNT}")
    print("All available CPU cores will be used.")
    print()

else:
    print()
    print("\033[42m CPU CORE CONFIGURATION CHECK PASSED \033[0m")
    print(f"Configured CPU_COUNT: {CONFIG_CPU_COUNT}")
    print(f"Detected logical CPU cores: {MAX_CPU_COUNT}")
    print(f"{CONFIG_CPU_COUNT} CPU cores will be used.")
    print()


###### RAM Check
MEM = psutil.virtual_memory()
TOTAL_RAM_GB = MEM.total / (1024 ** 3)
AVAILABLE_RAM_GB = MEM.available / (1024 ** 3)

if TOTAL_RAM_GB < 16:
    print()
    print(f"{YELLOW}[WARNING]{RESET} MEMORY CHECK")
    print(f"Total RAM      : {TOTAL_RAM_GB:.2f} GB")
    print("Recommended total RAM: 16 GB or more.")
else:
    print(f"{GREEN}[PASS]{RESET} TOTAL RAM CHECK")
    print(f"Total RAM      : {TOTAL_RAM_GB:.2f} GB")


if AVAILABLE_RAM_GB < 8:
    print()
    print(f"{RED}[FAIL]{RESET} AVAILABLE MEMORY CHECK")
    print(f"Available RAM  : {AVAILABLE_RAM_GB:.2f} GB")
    print("Minimum required available RAM: 8 GB")

    raise RuntimeError(
        f"Insufficient available RAM. Detected {AVAILABLE_RAM_GB:.2f} GB. Minimum required: 8 GB."
    )

print(f"{GREEN}[PASS]{RESET} AVAILABLE MEMORY CHECK")
print(f"Available RAM  : {AVAILABLE_RAM_GB:.2f} GB")