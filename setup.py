import setuptools


with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="golf",
    version="0.0.1",
    description="Golf",
    packages=setuptools.find_packages(),
    python_requires=">=3.6.0",
    install_requires=install_requires,
)
