"""
Process coverage data path
"""

BUILD_DIR = ""
with open('.cov/build-settings.txt', 'r') as in_file:
	for line in in_file:
		if 'PROJECT_TEMP_ROOT =' in line:
			raw_dir = line.strip().strip('\n').split('=')[-1].strip()
			print(raw_dir)
			BUILD_DIR = 
