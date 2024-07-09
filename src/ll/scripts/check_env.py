REQUIRED_PACKAGES = [
    "beartype",
    "cloudpickle",
    "jaxtyping",
    "lightning",
    "lightning_fabric",
    "lightning_utilities",
    "lovely_numpy",
    "lovely_tensors",
    "numpy",
    "psutil",
    "pydantic",
    "pydantic_core",
    "pysnooper",
    "rich",
    "tabulate",
    "torch",
    "torchmetrics",
    "tqdm",
    "typing_extensions",
    "wrapt",
    "yaml",
]


def main():
    import importlib.util
    import sys

    missing_packages: list[str] = []
    for package_name in REQUIRED_PACKAGES:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            missing_packages.append(package_name)

    if missing_packages:
        sys.exit(f"Error: Missing required packages: {', '.join(missing_packages)}")


if __name__ == "__main__":
    main()
