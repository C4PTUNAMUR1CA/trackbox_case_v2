from setuptools import setup, find_packages

setup(
    name="trackbox_case",
    version="0.0.1",
    description="A package to perform Football Analytics",
    author="Nikita Pavlov",
    author_email="nikita.pavlov.sva@hotmail.com",
    url="https://github.com/C4PTUNAMUR1CA/trackbox_case",
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.10",
)