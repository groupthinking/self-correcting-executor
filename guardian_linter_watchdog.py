import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration
WATCH_DIRECTORY = "."  # Watch the entire repository
PYTHON_EXTENSIONS = ('.py',)
PLACEHOLDER_STRINGS = ['TODO', 'FIXME', 'XXX', 'PLACEHOLDER']
MIN_TEST_COVERAGE = 80  # Minimum test coverage percentage

class GuardianAgent(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(PYTHON_EXTENSIONS):
            print(f"Guardian Agent: Detected change in {event.src_path}")
            self.run_linter(event.src_path)
            self.check_placeholders(event.src_path)
            self.run_test_coverage_analysis()

    def run_linter(self, file_path):
        """Runs a linter on the specified file."""
        print(f"Guardian Agent: Running linter on {file_path}...")
        try:
            subprocess.run(['pylint', file_path], check=True)
            print(f"Guardian Agent: Linter passed for {file_path}")
        except subprocess.CalledProcessError as e:
            print(f"Guardian Agent: Linter failed for {file_path}: {e}")
        except FileNotFoundError:
            print("Guardian Agent: pylint not found. Please install it with 'pip install pylint'")

    def check_placeholders(self, file_path):
        """Checks for placeholder strings in the specified file."""
        print(f"Guardian Agent: Checking for placeholders in {file_path}...")
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                for placeholder in PLACEHOLDER_STRINGS:
                    if placeholder in line:
                        print(f"Guardian Agent: Found placeholder '{placeholder}' in {file_path} at line {i+1}")

    def run_test_coverage_analysis(self):
        """Runs test coverage analysis for the entire project."""
        print("Guardian Agent: Running test coverage analysis...")
        try:
            subprocess.run(['coverage', 'run', '-m', 'pytest'], check=True)
            result = subprocess.run(['coverage', 'json'], capture_output=True, text=True, check=True)
            coverage_data = json.loads(result.stdout)
            coverage_percentage = int(coverage_data['totals']['percent'])
            if coverage_percentage < MIN_TEST_COVERAGE:
                print(f"Guardian Agent: Test coverage is {coverage_percentage}%, which is below the minimum of {MIN_TEST_COVERAGE}%")
            else:
                print(f"Guardian Agent: Test coverage is {coverage_percentage}%. Good job!")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Guardian Agent: Test coverage analysis failed: {e}")
            print("Guardian Agent: Please ensure you have 'coverage' and 'pytest' installed ('pip install coverage pytest')")


if __name__ == "__main__":
    event_handler = GuardianAgent()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=True)
    observer.start()
    print(f"Guardian Agent started. Watching directory: {WATCH_DIRECTORY}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
