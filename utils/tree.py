# from rich.tree import Tree
# from rich import print
# from pathlib import Path

# def build_tree(path: Path, tree: Tree, max_entries: int = 5, exclude_dirs=None):
#     if exclude_dirs is None:
#         exclude_dirs = set()

#     try:
#         entries = sorted(path.iterdir())
#     except PermissionError:
#         tree.add("[red][Permission Denied]")
#         return

#     # Filter out excluded folders
#     entries = [e for e in entries if e.name not in exclude_dirs]
#     display_entries = entries[:max_entries]

#     for item in display_entries:
#         if item.is_dir():
#             branch = tree.add(f"[bold blue]{item.name}/")
#             build_tree(item, branch, max_entries, exclude_dirs)
#         else:
#             tree.add(item.name)

#     if len(entries) > max_entries:
#         tree.add(f"[dim]... and {len(entries) - max_entries} more")

# # Example usage
# if __name__ == "__main__":
#     exclude = {"__pycache__", ".git", ".venv_3.12", "Final_Test", "Training"}
#     root_path = Path(".")
#     tree = Tree(f"[bold green]{root_path.resolve().name}")
#     build_tree(root_path, tree, max_entries=14, exclude_dirs=exclude)
#     print(tree)



from rich.tree import Tree
from rich import print
from pathlib import Path

def build_tree(path: Path, tree: Tree, max_entries: int = 5, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = set()

    try:
        entries = sorted(
            path.iterdir(),
            key=lambda e: (not e.is_dir(), e.name.lower())  # Folders first, then files (case-insensitive)
        )
    except PermissionError:
        tree.add("[red][Permission Denied]")
        return

    # Filter out excluded folders
    entries = [e for e in entries if e.name not in exclude_dirs]
    display_entries = entries[:max_entries]

    for item in display_entries:
        if item.is_dir():
            branch = tree.add(f"[bold blue]{item.name}/")
            build_tree(item, branch, max_entries, exclude_dirs)
        else:
            tree.add(item.name)

    if len(entries) > max_entries:
        tree.add(f"[dim]... and {len(entries) - max_entries} more")

# Example usage
if __name__ == "__main__":
    exclude = {"__pycache__", ".git", ".venv_3.12", "Final_Test", "Training"}
    root_path = Path(".")
    tree = Tree(f"[bold green]{root_path.resolve().name}")
    print("Excluded:", exclude)
    build_tree(root_path, tree, max_entries=14, exclude_dirs=exclude)
    print(tree)