from setuptools import setup, find_packages

setup(
    name="lbh",
    version="1.0.0",
    description="Level Based Hacking Environment",
    author="LBH Project",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    install_requires=["numpy", "gymnasium", "pyglet<2", "six"],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)

