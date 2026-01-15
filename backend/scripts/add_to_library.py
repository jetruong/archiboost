#!/usr/bin/env python3
"""
Script to manually add files to the library from the backend.

Usage:
    python scripts/add_to_library.py <file_path> [--display-name NAME] [--tags TAG1,TAG2] [--description DESC]
    python scripts/add_to_library.py --directory <dir_path> [--recursive] [--tags TAG1,TAG2]
    python scripts/add_to_library.py --list
    python scripts/add_to_library.py --delete <file_id>

Examples:
    # Add a single file
    python scripts/add_to_library.py Pair\ 1/Sheet-001.pdf --display-name "Water Closet Detail" --tags "bathroom,accessible"
    
    # Add all PDFs from a directory
    python scripts/add_to_library.py --directory "Pair 1" --tags "test,example"
    
    # List all files in library
    python scripts/add_to_library.py --list
    
    # Delete a file from library
    python scripts/add_to_library.py --delete <file_id>
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.library import library_service


async def add_file(
    file_path: Path,
    display_name: str = None,
    tags: list[str] = None,
    description: str = None,
) -> None:
    """Add a single file to the library."""
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    
    if not file_path.is_file():
        print(f"Error: Path is not a file: {file_path}", file=sys.stderr)
        sys.exit(1)
    
    # Read file content
    content = file_path.read_bytes()
    
    # Determine content type
    content_type = None
    if file_path.suffix.lower() == ".png":
        content_type = "image/png"
    elif file_path.suffix.lower() == ".pdf":
        content_type = "application/pdf"
    
    # Add to library
    try:
        library_file = await library_service.add_file(
            content=content,
            filename=file_path.name,
            content_type=content_type,
            display_name=display_name or file_path.stem,
            tags=tags or [],
            description=description,
        )
        
        print(f"✓ Added file to library:")
        print(f"  ID: {library_file.id}")
        print(f"  Name: {library_file.display_name}")
        print(f"  Type: {library_file.file_type}")
        print(f"  Size: {library_file.size_bytes / 1024 / 1024:.2f} MB")
        print(f"  Path: {library_file.storage_path}")
        
    except Exception as e:
        print(f"Error adding file: {e}", file=sys.stderr)
        sys.exit(1)


async def add_directory(
    dir_path: Path,
    recursive: bool = False,
    tags: list[str] = None,
    file_extensions: list[str] = None,
) -> None:
    """Add all matching files from a directory to the library."""
    if not dir_path.exists():
        print(f"Error: Directory not found: {dir_path}", file=sys.stderr)
        sys.exit(1)
    
    if not dir_path.is_dir():
        print(f"Error: Path is not a directory: {dir_path}", file=sys.stderr)
        sys.exit(1)
    
    # Default extensions
    if file_extensions is None:
        file_extensions = [".pdf", ".png"]
    
    # Find files
    if recursive:
        files = [
            f for f in dir_path.rglob("*")
            if f.is_file() and f.suffix.lower() in file_extensions
        ]
    else:
        files = [
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in file_extensions
        ]
    
    if not files:
        print(f"No matching files found in {dir_path}")
        return
    
    print(f"Found {len(files)} file(s) to add...")
    
    success_count = 0
    error_count = 0
    
    for file_path in files:
        try:
            await add_file(
                file_path=file_path,
                display_name=None,  # Use filename
                tags=tags or [],
                description=None,
            )
            success_count += 1
        except Exception as e:
            print(f"✗ Failed to add {file_path.name}: {e}", file=sys.stderr)
            error_count += 1
    
    print(f"\nSummary: {success_count} added, {error_count} failed")


def list_files() -> None:
    """List all files in the library."""
    files = library_service.list_files()
    
    if not files:
        print("Library is empty.")
        return
    
    print(f"Library contains {len(files)} file(s):\n")
    
    for file in files:
        print(f"ID: {file.id}")
        print(f"  Name: {file.display_name}")
        print(f"  Filename: {file.filename}")
        print(f"  Type: {file.file_type}")
        print(f"  Size: {file.size_bytes / 1024 / 1024:.2f} MB")
        print(f"  Uploaded: {file.uploaded_at}")
        if file.tags:
            print(f"  Tags: {', '.join(file.tags)}")
        print()


def delete_file(file_id: str) -> None:
    """Delete a file from the library."""
    file = library_service.get_file(file_id)
    if not file:
        print(f"Error: File not found: {file_id}", file=sys.stderr)
        sys.exit(1)
    
    success = library_service.delete_file(file_id)
    if success:
        print(f"✓ Deleted file: {file.display_name} ({file_id})")
    else:
        print(f"Error: Failed to delete file: {file_id}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Manage the file library from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Main action
    parser.add_argument(
        "file_path",
        nargs="?",
        type=Path,
        help="Path to file to add (or use --directory, --list, --delete)",
    )
    
    # Options for adding files
    parser.add_argument(
        "--display-name",
        type=str,
        help="Display name for the file (default: filename without extension)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        help="Comma-separated tags (e.g., 'bathroom,accessible,test')",
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Description of the file",
    )
    
    # Directory mode
    parser.add_argument(
        "--directory",
        "-d",
        type=Path,
        help="Add all matching files from a directory",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively search subdirectories (use with --directory)",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        help="Comma-separated file extensions to include (default: pdf,png)",
    )
    
    # List mode
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all files in the library",
    )
    
    # Delete mode
    parser.add_argument(
        "--delete",
        type=str,
        help="Delete a file from the library by ID",
    )
    
    args = parser.parse_args()
    
    # Parse tags
    tags = None
    if args.tags:
        tags = [t.strip() for t in args.tags.split(",")]
    
    # Parse extensions
    extensions = None
    if args.extensions:
        extensions = [f".{ext.strip().lstrip('.')}" for ext in args.extensions.split(",")]
    
    # Execute action
    if args.delete:
        delete_file(args.delete)
    elif args.list:
        list_files()
    elif args.directory:
        asyncio.run(add_directory(
            dir_path=args.directory,
            recursive=args.recursive,
            tags=tags,
            file_extensions=extensions,
        ))
    elif args.file_path:
        asyncio.run(add_file(
            file_path=args.file_path,
            display_name=args.display_name,
            tags=tags,
            description=args.description,
        ))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
