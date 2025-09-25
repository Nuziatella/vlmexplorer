#!/usr/bin/env python3
"""
Python Cache Cleaner for Virtual Environment
Clears __pycache__ directories and pip cache within a project/venv.
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

class PythonCacheCleaner:
    def __init__(self, project_path=".", verbose=True, dry_run=False):
        self.project_path = Path(project_path).resolve()
        self.verbose = verbose
        self.dry_run = dry_run
        self.cleaned_size = 0
        self.errors = []
    
    def log(self, message):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[INFO] {message}")
    
    def log_error(self, message):
        """Log error message."""
        error_msg = f"[ERROR] {message}"
        self.errors.append(error_msg)
        if self.verbose:
            print(error_msg)
    
    def format_size(self, size_bytes):
        """Format bytes to human readable format."""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f} {size_names[i]}"
    
    def get_dir_size(self, path):
        """Get total size of directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, IOError):
                        continue
        except (OSError, IOError):
            pass
        return total_size
    
    def clean_pycache(self):
        """Clean all __pycache__ directories in the project."""
        self.log("Cleaning __pycache__ directories...")
        
        pycache_dirs = list(self.project_path.rglob("__pycache__"))
        cleaned_count = 0
        pycache_size = 0
        
        if not pycache_dirs:
            self.log("No __pycache__ directories found")
            return 0
        
        for pycache_dir in pycache_dirs:
            try:
                size = self.get_dir_size(pycache_dir)
                pycache_size += size
                
                if self.dry_run:
                    self.log(f"Would remove: {pycache_dir} ({self.format_size(size)})")
                else:
                    shutil.rmtree(pycache_dir)
                    self.log(f"Removed: {pycache_dir} ({self.format_size(size)})")
                
                cleaned_count += 1
                
            except (OSError, IOError, PermissionError) as e:
                self.log_error(f"Failed to remove {pycache_dir}: {e}")
        
        self.cleaned_size += pycache_size
        self.log(f"__pycache__ cleanup: {cleaned_count} directories, {self.format_size(pycache_size)}")
        return cleaned_count
    
    def clean_pyc_files(self):
        """Clean individual .pyc and .pyo files that might be outside __pycache__."""
        self.log("Cleaning .pyc and .pyo files...")
        
        pyc_files = list(self.project_path.rglob("*.pyc")) + list(self.project_path.rglob("*.pyo"))
        cleaned_count = 0
        pyc_size = 0
        
        if not pyc_files:
            self.log("No .pyc/.pyo files found")
            return 0
        
        for pyc_file in pyc_files:
            try:
                size = pyc_file.stat().st_size
                pyc_size += size
                
                if self.dry_run:
                    self.log(f"Would remove: {pyc_file} ({self.format_size(size)})")
                else:
                    pyc_file.unlink()
                    self.log(f"Removed: {pyc_file} ({self.format_size(size)})")
                
                cleaned_count += 1
                
            except (OSError, IOError, PermissionError) as e:
                self.log_error(f"Failed to remove {pyc_file}: {e}")
        
        self.cleaned_size += pyc_size
        self.log(f".pyc/.pyo cleanup: {cleaned_count} files, {self.format_size(pyc_size)}")
        return cleaned_count
    
    def get_pip_cache_info(self):
        """Get pip cache directory and size info."""
        try:
            # Try to get pip cache dir
            result = subprocess.run([sys.executable, "-m", "pip", "cache", "dir"], 
                                  capture_output=True, text=True, check=True)
            cache_dir = Path(result.stdout.strip())
            
            if cache_dir.exists():
                cache_size = self.get_dir_size(cache_dir)
                return cache_dir, cache_size
            else:
                return None, 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log_error("Could not determine pip cache directory")
            return None, 0
    
    def clean_pip_cache(self):
        """Clean pip cache using pip cache purge."""
        self.log("Cleaning pip cache...")
        
        # Get cache info first
        cache_dir, cache_size = self.get_pip_cache_info()
        
        if cache_size == 0:
            self.log("No pip cache found or cache is empty")
            return False
        
        self.log(f"Pip cache location: {cache_dir}")
        self.log(f"Pip cache size: {self.format_size(cache_size)}")
        
        if self.dry_run:
            self.log(f"Would clear pip cache ({self.format_size(cache_size)})")
            self.cleaned_size += cache_size
            return True
        
        try:
            # Use pip cache purge to clear the cache
            result = subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], 
                                  capture_output=True, text=True, check=True)
            
            self.log("Successfully cleared pip cache")
            self.log(f"Pip output: {result.stdout.strip()}")
            self.cleaned_size += cache_size
            return True
            
        except subprocess.CalledProcessError as e:
            self.log_error(f"Failed to clear pip cache: {e.stderr}")
            return False
        except FileNotFoundError:
            self.log_error("pip command not found")
            return False
    
    def analyze_caches(self):
        """Analyze cache sizes without cleaning."""
        self.log(f"Analyzing Python caches in: {self.project_path}")
        
        # Analyze __pycache__ directories
        pycache_dirs = list(self.project_path.rglob("__pycache__"))
        pycache_size = sum(self.get_dir_size(d) for d in pycache_dirs)
        
        # Analyze .pyc/.pyo files
        pyc_files = list(self.project_path.rglob("*.pyc")) + list(self.project_path.rglob("*.pyo"))
        pyc_size = sum(f.stat().st_size for f in pyc_files if f.exists())
        
        # Analyze pip cache
        pip_cache_dir, pip_cache_size = self.get_pip_cache_info()
        
        print("\nPython Cache Analysis")
        print("=" * 40)
        print(f"__pycache__ directories: {len(pycache_dirs)} ({self.format_size(pycache_size)})")
        print(f".pyc/.pyo files: {len(pyc_files)} ({self.format_size(pyc_size)})")
        print(f"Pip cache: {self.format_size(pip_cache_size)}")
        if pip_cache_dir:
            print(f"  Location: {pip_cache_dir}")
        
        total_size = pycache_size + pyc_size + pip_cache_size
        print(f"\nTotal cache size: {self.format_size(total_size)}")
    
    def clean_all(self):
        """Clean all Python caches."""
        self.log(f"Starting Python cache cleanup in: {self.project_path}")
        if self.dry_run:
            self.log("DRY RUN MODE - No files will be actually deleted")
        
        total_cleaned = 0
        
        # Clean __pycache__ directories
        total_cleaned += self.clean_pycache()
        
        # Clean individual .pyc/.pyo files
        total_cleaned += self.clean_pyc_files()
        
        # Clean pip cache
        if self.clean_pip_cache():
            total_cleaned += 1
        
        return total_cleaned

def display_menu():
    """Display the interactive menu options."""
    print("\n" + "=" * 60)
    print("          Python Cache Cleaner")
    print("=" * 60)
    print("1. Clean all caches (__pycache__ + pip cache)")
    print("2. Clean only __pycache__ directories")
    print("3. Clean only pip cache")
    print("4. Analyze cache sizes (no cleaning)")
    print("5. Dry run - show what would be cleaned")
    print("6. Exit")
    print("-" * 60)

def get_user_choice():
    """Get user's menu choice."""
    while True:
        try:
            choice = input("Select an option (1-6): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6']:
                return int(choice)
            else:
                print("Invalid choice. Please enter a number between 1-6.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        except EOFError:
            print("\nExiting...")
            sys.exit(0)

def get_project_path():
    """Get project path from user."""
    path_input = input("Enter project path (press Enter for current directory): ").strip()
    return path_input if path_input else "."

def interactive_mode():
    """Run the script in interactive mode."""
    print("Welcome to Python Cache Cleaner!")
    
    # Get project path
    project_path = get_project_path()
    
    while True:
        display_menu()
        choice = get_user_choice()
        
        # Create cleaner instance based on choice
        dry_run = (choice == 5)
        cleaner = PythonCacheCleaner(
            project_path=project_path,
            verbose=True,
            dry_run=dry_run
        )
        
        try:
            if choice == 1:  # Clean all
                print(f"\nCleaning all caches in: {cleaner.project_path}")
                cleaner.clean_all()
                
            elif choice == 2:  # Clean __pycache__ only
                print(f"\nCleaning __pycache__ directories in: {cleaner.project_path}")
                cleaner.clean_pycache() + cleaner.clean_pyc_files()
                
            elif choice == 3:  # Clean pip cache only
                print("\nCleaning pip cache...")
                cleaner.clean_pip_cache()
                
            elif choice == 4:  # Analyze
                cleaner.analyze_caches()
                input("\nPress Enter to continue...")
                continue
                
            elif choice == 5:  # Dry run
                print(f"\nDry run - showing what would be cleaned in: {cleaner.project_path}")
                cleaner.clean_all()
                
            elif choice == 6:  # Exit
                print("Goodbye!")
                break
            if choice in [1, 2, 3, 5]:
                print("\nOperation completed!")
                print(f"Total space {'would be ' if dry_run else ''}freed: {cleaner.format_size(cleaner.cleaned_size)}")
                
                if cleaner.errors:
                    print(f"\nErrors encountered: {len(cleaner.errors)}")
                    for error in cleaner.errors:
                        print(error)
                
                input("\nPress Enter to continue...")
        
        except Exception as e:
            print(f"Error during operation: {e}")
            input("\nPress Enter to continue...")

def main():
    parser = argparse.ArgumentParser(description="Clean Python caches (__pycache__ and pip cache)")
    parser.add_argument("path", nargs="?", default=None, 
                       help="Project path (default: interactive mode)")
    parser.add_argument("--dry-run", "-n", action="store_true", 
                       help="Show what would be deleted without actually deleting")
    parser.add_argument("--quiet", "-q", action="store_true", 
                       help="Suppress verbose output")
    parser.add_argument("--analyze", "-a", action="store_true", 
                       help="Analyze cache sizes without cleaning")
    parser.add_argument("--pycache-only", action="store_true", 
                       help="Only clean __pycache__ directories, skip pip cache")
    parser.add_argument("--pip-only", action="store_true", 
                       help="Only clean pip cache, skip __pycache__ directories")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode with menu")
    
    args = parser.parse_args()
    
    # If no path provided or interactive flag is set, run interactive mode
    if args.path is None or args.interactive:
        interactive_mode()
        return
    
    # Command-line mode
    cleaner = PythonCacheCleaner(
        project_path=args.path,
        verbose=not args.quiet,
        dry_run=args.dry_run
    )
    
    if args.analyze:
        cleaner.analyze_caches()
        return
    
    try:
        if args.pycache_only:
            cleaner.clean_pycache() + cleaner.clean_pyc_files()
        elif args.pip_only:
            cleaner.clean_pip_cache()
        else:
            cleaner.clean_all()
        
        print("\nCleanup completed!")
        print(f"Total space {'would be ' if args.dry_run else ''}freed: {cleaner.format_size(cleaner.cleaned_size)}")
        
        if cleaner.errors:
            print(f"\nErrors encountered: {len(cleaner.errors)}")
            for error in cleaner.errors:
                print(error)
    
    except KeyboardInterrupt:
        print("\nCleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during cleanup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()