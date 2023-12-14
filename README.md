# Fractal Image Compression

Requirements:
- python>=3.10

Installation:
- `pip install -r requirements.txt`



# Theoretical background

1. Copy machine, fractals and iterated function systems

2. Contractive mapping theorem

3. Fractal image encoding and decoding (abstractly)


# Quadtree algorithm

1. pseudocode / description

2. complexity analysis

3. more interesting implementation details

# HV algorithm

same as above

# Evaluation

1. comparison (between uncompressed - jpeg - quadtree - hv) for few selected images (cauliflower, lena, something with mixed self similar and not same similar structure) 
- display images for high fidelity parameters
- display images for high compression parameters 

2. Quadtree
- psnr(decoding iteration) plot
- compression_rate(psnr) plot for selected parameters

3. HV, same as above
