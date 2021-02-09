from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = 'magpylib3',
    version = '3.0.0',
    author = 'Michael Ortner & Friends',
    author_email = 'magpylib@gmail.com',
    description = 'Free Python package for magnetic field computation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license = 'AGPLv3+',
    packages = find_packages(),
    install_requires=[
          'numpy>=1.16',
          'matplotlib>=3.1',
      ],
    classifiers=[
        'Development Status :: 5 Stable',
        'Programming Language :: Python :: 3.9',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
    ],
    python_requires = '>=3.6',
    keywords = 'magnet current field magnetism physics',
)