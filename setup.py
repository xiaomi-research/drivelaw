"""
Setup script for DriveLaW project.
This installs all dependencies for video generation and planning components.
"""
import os
from pathlib import Path
from setuptools import setup, find_packages

# Try to import packaging for version comparison, but don't fail if not available
try:
    from packaging import version
    HAS_PACKAGING = True
except ImportError:
    HAS_PACKAGING = False

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

# Get the base directory
base_dir = Path(__file__).parent

# Read requirements from sub-components
def read_requirements(file_path):
    """Read requirements from a file."""
    requirements = []
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith("#"):
                    # Handle git+ URLs and other special cases
                    requirements.append(line)
    return requirements

# Collect all dependencies
all_dependencies = []

# 1. Video Generation Inference dependencies (from pyproject.toml)
video_gen_deps = [
    "torch>=2.1.0",
    "diffusers>=0.28.2",
    "transformers>=4.47.2,<4.52.0",
    "sentencepiece>=0.1.96",
    "huggingface-hub~=0.30",
    "einops",
    "timm",
    # Inference extras
    "imageio[ffmpeg]",
    "av",
    "torchvision",
]

# 2. Video Generation Training dependencies (from pyproject.toml)
video_train_deps = [
    "accelerate>=1.2.1",
    "av>=14.2.1",
    "bitsandbytes>=0.45.2; sys_platform == 'linux'",
    "decord>=0.6.0; sys_platform == 'linux'",
    "diffusers>=0.32.1",
    "gradio==5.33.0",
    "imageio>=2.37.0",
    "imageio-ffmpeg>=0.6.0",
    "opencv-python>=4.11.0.86",
    "optimum-quanto>=0.2.6",
    "pandas>=2.2.3",
    "peft>=0.14.0",
    "pillow-heif>=0.21.0",
    "protobuf>=5.29.3",
    "pydantic>=2.10.4",
    "rich>=13.9.4",
    "safetensors>=0.5.0",
    "scenedetect>=0.6.5.2",
    "sentencepiece>=0.2.0",
    "setuptools>=75.6.0",
    "torch>=2.6.0",
    "torchvision>=0.21.0",
    "typer>=0.15.1",
    "wandb>=0.19.11",
]

# 3. Planner dependencies (from recogdrive-main/requirements.txt)
planner_deps = [
    "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2",
    "positional-encodings==6.0.1",
    "aioboto3",
    "aiofiles",
    "bokeh==2.4.3",
    "control==0.9.1",
    "Fiona",
    "geopandas>=0.12.1",
    "guppy3==3.1.2",
    "hydra-core==1.2.0",
    "joblib",
    "matplotlib",
    "nest_asyncio",
    "numpy==1.26.4",
    "opencv-python==4.9.0.80",
    "pandas",
    "Pillow",
    "psutil",
    "pyarrow",
    "pyinstrument",
    "pyogrio",
    "pyquaternion>=0.9.5",
    "rasterio",
    "ray",
    "retry",
    "rtree",
    "scipy",
    "selenium",
    "setuptools==65.5.1",
    "Shapely>=2.0.0",
    "SQLAlchemy==1.4.27",
    "sympy",
    "tornado",
    "tqdm",
    "ujson",
    "diffusers",
    "torch",
    "torchvision",
    "pytorch-lightning",
    "tensorboard==2.16.2",
    "protobuf==4.25.3",
    "notebook",
    "timm",
]

# Merge all dependencies and resolve version conflicts
def merge_dependencies(*dep_lists):
    """Merge multiple dependency lists, resolving version conflicts."""
    dep_dict = {}
    
    def extract_package_name(dep_string):
        """Extract package name from dependency string."""
        # Handle git+ URLs
        if "@ git+" in dep_string:
            return dep_string.split("@ git+")[0].strip()
        # Handle version specifiers
        for op in [">=", "==", "~=", "<=", ">", "<", "@"]:
            if op in dep_string:
                return dep_string.split(op)[0].strip()
        return dep_string.strip()
    
    for deps in dep_lists:
        for dep in deps:
            dep = dep.strip()
            if not dep:
                continue
                
            # Extract package name
            if ";" in dep:
                # Platform-specific dependency (e.g., "package>=1.0; sys_platform == 'linux'")
                name_part, marker = dep.split(";", 1)
                name = extract_package_name(name_part)
                # Keep platform-specific dependencies as-is
                if name not in dep_dict or ";" not in dep_dict[name]:
                    dep_dict[name] = dep
            else:
                name = extract_package_name(dep)
                
                # Handle version conflicts
                if name in dep_dict:
                    existing = dep_dict[name]
                    # If existing has platform marker, keep it
                    if ";" in existing:
                        continue  # Keep platform-specific one
                    
                    # Prefer exact versions (==)
                    if "==" in dep:
                        dep_dict[name] = dep
                    elif "==" in existing:
                        continue  # Keep existing exact version
                    # Prefer higher minimum versions (>=)
                    elif ">=" in dep and ">=" in existing:
                        # Extract versions and compare
                        try:
                            if HAS_PACKAGING:
                                dep_ver = dep.split(">=")[1].split(",")[0].split(";")[0].strip()
                                exist_ver = existing.split(">=")[1].split(",")[0].split(";")[0].strip()
                                # Use the higher version requirement
                                if version.parse(dep_ver) >= version.parse(exist_ver):
                                    dep_dict[name] = dep
                            else:
                                # Fallback: keep the longer string (more specific)
                                dep_dict[name] = dep if len(dep) > len(existing) else existing
                        except:
                            # If version parsing fails, keep the longer string (more specific)
                            dep_dict[name] = dep if len(dep) > len(existing) else existing
                    else:
                        # Default: keep the new one (or more specific)
                        dep_dict[name] = dep if len(dep) >= len(existing) else existing
                else:
                    dep_dict[name] = dep
    
    return list(dep_dict.values())

all_dependencies = merge_dependencies(video_gen_deps, video_train_deps, planner_deps)

# Sort dependencies for better readability (but keep git+ URLs first)
git_deps = [d for d in all_dependencies if d.startswith("git+") or "@ git+" in d]
other_deps = [d for d in all_dependencies if not (d.startswith("git+") or "@ git+" in d)]
other_deps.sort()
all_dependencies = git_deps + other_deps

setup(
    name="drivelaw",
    version="1.0.0",
    description="DriveLaW: Driving Video Generation and Planning with Latent Diffusion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DriveLaW Team",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=all_dependencies + (["packaging>=20.0"] if not HAS_PACKAGING else []),
    extras_require={
        "dev": [
            "pytest",
            "pre-commit>=4.0.1",
            "ruff>=0.8.6",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    zip_safe=False,
)
