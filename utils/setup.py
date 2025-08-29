from setuptools import setup, find_packages

setup(
    name="utils",
    version="0.1.0",
    description="Utility modules for the monorepo, including logging and more.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "certifi==2025.8.3",
        "sentry-sdk==2.35.1",
        "urllib3==2.5.0",
    ],
    python_requires=">=3.6",
)
