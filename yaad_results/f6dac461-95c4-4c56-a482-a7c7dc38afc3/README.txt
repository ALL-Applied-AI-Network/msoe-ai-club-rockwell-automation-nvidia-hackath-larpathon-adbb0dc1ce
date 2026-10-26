This directory contains the cancelled job metadata and complete log tail returned by the YAAD API.

The preserved remote directory is ready at:
  /workspace/results/f6dac461-95c4-4c56-a482-a7c7dc38afc3

The full SCP transfer was attempted on 2026-10-25 but the private YAAD address timed out because this computer was not connected to the CSI network/VPN. After connecting FortiClient, run from yaad_results/:
  scp -o BatchMode=yes -r yt-larpathon78f7@172.30.102.93:results/f6dac461-95c4-4c56-a482-a7c7dc38afc3 .
