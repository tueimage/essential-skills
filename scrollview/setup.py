import setuptools


setuptools.setup(
    name="slycer",
    version="0.0.1",
    author="Koen A. J. Eppenhof",
    author_email="k.a.j.eppenhof@tue.nl",
    description="ScrollView objects for plotting 3D images in standard Matplotlib axes with scrolling support",
    long_description="""ScrollView objects for plotting 3D images in standard Matplotlib axes with scrolling support""",
    long_description_content_type="text/markdown",
    url="https://github.com/tueimage/scrollview",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
