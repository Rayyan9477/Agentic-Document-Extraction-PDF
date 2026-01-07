"""Kill all processes listening on port 8000."""
import re
import subprocess


# Get netstat output
result = subprocess.run(["netstat", "-ano"], check=False, capture_output=True, text=True)

# Extract PIDs listening on port 8000
pids = set(re.findall(r":8000.*?LISTENING\s+(\d+)", result.stdout))
#Extract pIDs for listening ports on 3000
pids_3000 = set(re.findall(r":3000.*?LISTENING\s+(\d+)", result.stdout))
# now kill both pids
pids.update(pids_3000)


# Kill each PID
for pid in pids:
    if pid != "0":
        try:
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True, check=False)
        except Exception as e:
            pass

