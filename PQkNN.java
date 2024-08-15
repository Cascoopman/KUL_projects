/**
 * Copyright (c) DTAI - KU Leuven - All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Maaike Van Roy
 */
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * To use the provided KMeans implementation, add the jar to the build path, 
 * import like this and use like any other class.
 */
import kmeans.KMeans; 


public class PQkNN {
	
	private int n; // Amount of subvectors
	private int k; // k of k-Means
	private int[] trainlabels;
	private int[][] compressedData;
	private ArrayList<double[][]> subvectorCentroids;
	private double[][] distances;
	private int subvectorLength;
	
	/**
	 * Contruct a new instance of kNN with Product Quantization
	 * @param n Amount of subvectors
	 * @param c Determines the amount of clusters for KMeans, i.e., k = 2**c
	 */
	public PQkNN(int n, int c) {
		this.n = n;
		this.k = 1 << c;
	}
	
	
	/**
	 * Compress the given training examples via the product quantization method (see assignment for paper and blog post).
	 * The necessary data structures are to be stored within class instantiation.
	 * @param traindata The training examples, 2D integer matrix where each row represents an image.
	 * @param trainlabels Labels for the training examples, 0..9
	 */
	public void compress(int[][] traindata, int[] trainlabels) {
        this.trainlabels = trainlabels;
        int numSamples = traindata.length;
        int featureLength = traindata[0].length;
		KMeans km = new KMeans(numSamples, this.k);

        this.subvectorLength = featureLength / this.n;
        this.compressedData = new int[numSamples][this.n];
        this.subvectorCentroids = new ArrayList<>(this.n);

        int remainder = featureLength % this.n;

		// Compress each subvector by applying k-Means
        for (int i = 0; i < this.n; i++) {
            int currentSubvectorLength = (i == this.n - 1 && remainder != 0) // Ensures proper handling of feature vector overflow
                                         ? featureLength - i * this.subvectorLength 
                                         : this.subvectorLength;

            int[][] subVectors = new int[numSamples][currentSubvectorLength];
            for (int j = 0; j < numSamples; j++) {
                System.arraycopy(traindata[j], i * this.subvectorLength, subVectors[j], 0, currentSubvectorLength);
            }

            double[][] centroids = km.fit(subVectors);

            for (int j = 0; j < numSamples; j++) {
                this.compressedData[j][i] = km.predict(subVectors[j], centroids);
            }

            this.subvectorCentroids.add(centroids);
        }
	}

	/**
	 * Predicts the label of a given 1D-image example based on the PQkNN algorithm.
	 * @param testsample The given out-of-sample 1D image.
	 * @param nearestNeighbors k in kNN
	 * @return test image classification (0..9)
	 */
	public int predict(int[] testsample, int nearestNeighbors) {
		this.distances = new double[this.k][this.n];
		int maxLabel = 10; 
		int[] labelCount = new int[maxLabel + 1];
	
		// Generate Lookup Table
		for (int i = 0; i < this.n; i++) {
			int[] sampleSubvector = Arrays.copyOfRange(
				testsample, 
				i * this.subvectorLength, // Start of subvector
				(i == this.n - 1) ? testsample.length : (i + 1) * this.subvectorLength // Ending of subvector (Proper handling of overflow)
				);
			double[][] centroids = subvectorCentroids.get(i);
	
			for (int j = 0; j < this.k; j++) {
				this.distances[j][i] = calculateDistance(sampleSubvector, centroids[j]);
			}
		}
	
		// Determine the k-nearest neighbors of the sample
		PriorityQueue<int[]> maxHeap = new PriorityQueue<>(nearestNeighbors, new Comparator<int[]>() {
			public int compare(int[] a, int[] b) {
				return Double.compare(b[0], a[0]); // Max-heap to maintain k smallest distances, prevents need for sorting
			}
		});
		for (int i = 0; i < this.compressedData.length; i++) {
			double sum = 0.0;
			for (int j = 0; j < this.n; j++) {
				int centroid_id = this.compressedData[i][j];
				sum += this.distances[centroid_id][j];
			}
	
			if (maxHeap.size() < nearestNeighbors) {
				maxHeap.offer(new int[]{(int) sum, i}); // Store the distance (sum) and the index (i) of the training example
			} else if (sum < maxHeap.peek()[0]) {
				maxHeap.poll();
				maxHeap.offer(new int[]{(int) sum, i});
			}
		}
	
		// Retrieve most common label from the k-nearest neighbors
		int mostCommonLabel = 0;
		while (!maxHeap.isEmpty()) {
			int[] pair = maxHeap.poll();
			int label = this.trainlabels[pair[1]];
			labelCount[label]++;
		
			if (labelCount[label] > labelCount[mostCommonLabel]) {
				mostCommonLabel = label;
			}
		}
		
		return mostCommonLabel;
	}
	
	/**
	 * Calculate the distance between the given example and a centroid.
	 * @param example1 The given example.
	 * @param example2 The given centroid.
	 * @return The distance.
	 */
	private static double calculateDistance(int[] example1, double[] example2) {
		double sum = 0.0;
		int length = example1.length;
	
		for (int i = 0; i < length; i++) {
			sum += Math.pow((double)example1[i] - example2[i], 2.0);
		}

		return sum;
	}

}

