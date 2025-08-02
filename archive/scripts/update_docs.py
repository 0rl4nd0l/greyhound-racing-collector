import os
import subprocess

# Path to the changelog file and architecture diagrams
diff_path = 'docs/CHANGELOG.md'


def update_changelog(diff_output):
    with open(diff_path, 'a') as changelog:
        changelog.write('### Recent Changes\n')
        for line in diff_output.split('\n'):
            if line.startswith('def ') or line.startswith('class '):
                item = line.split(' ')[1].split('(')[0]  # Extract the function/class name
                changelog.write(f'- Modified: {item}\n')


def update_diagrams():
    # Logic to update architecture diagram links can be implemented here
    pass  # This function is left empty for now


def main():
    try:
        # Get the git diff for recent changes
        diff_output = subprocess.check_output(['git', 'diff', 'HEAD~1', 'HEAD', '--', '*.py'], universal_newlines=True)
        
        # Update the CHANGELOG.md
        update_changelog(diff_output)
        
        # Update architecture diagrams if needed
        update_diagrams()
        
        print('Documentation updated successfully.')
    except subprocess.CalledProcessError as e:
        print(f'Error while executing git diff: {e}')


if __name__ == '__main__':
    main()
