import setuptools

setuptools.setup(
    name="difference_finder",
    version="0.0.2",
    license='MIT',
    author="Jeongsol Kim",
    author_email="wjdthf3927@gmail.com",
    description="Find difference between two images using pytorch.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/jeongsol-kim/difference_finder",
    packages=setuptools.find_packages(),
    classifiers=[
        # 패키지에 대한 태그
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)