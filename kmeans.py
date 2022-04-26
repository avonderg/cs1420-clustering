"""
    This is a file you will have to fill in.
    It contains helper functions required by K-means method via iterative improvement
"""
import numpy as np
from random import sample

def init_centroids(k, inputs):
    """
    Selects k random rows from inputs and returns them as the chosen centroids
    Hint: use random.sample (it is already imported for you!)
    :param k: number of cluster centroids
    :param inputs: a 2D Numpy array, each row of which is one input
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    
    indices = range(len(inputs))

    selected_indices = sample(indices, k) #samples from that list of indices
    
    centroids = []
    # m = np.shape(inputs)[0]

    for i in range(k):
        # r = np.random.randint(0, m-1)
        centroids.append(inputs[selected_indices[i]])

    # print(np.array(centroids).shape)
    return np.array(centroids)


def assign_step(inputs, centroids):
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance
    :param inputs: inputs of data, a 2D Numpy array
    :param centroids: a Numpy array of k current centroids
    :return: a Numpy array of centroid indices, one for each row of the inputs
    """
    # TODO

    indices = []

    for j in range(len(inputs)): #i is index, j is corresponding vector
        subtracted = np.subtract(centroids, inputs[j])

        distance = np.linalg.norm(subtracted, axis=1)
        centroid = np.argmin(distance) #current centroid
        indices.append(centroid)
        # for centroid in centroids:
        #     dist = np.linalg.norm(np.array(j-centroid), axis=1) #difference of two vectors + get norm
        #     distances.append(dist)
        # closest_centroid_index =  min(range(len(distances)), key=lambda x: distances[x])
        # indices.append(closest_centroid_index)
    return np.array(indices)


def update_step(inputs, indices, k):
    """
    Computes the centroid for each cluster
    :param inputs: inputs of data, a 2D Numpy array
    :param indices: a Numpy array of centroid indices, one for each row of the inputs
    :param k: number of cluster centroids, an int
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    # enumerate all inputs, add each input to matrix according to their corresponding index
    # matrix to represent new centroids in shape of k , 64 -> return after function
    matrix = np.zeros((k, inputs.shape[1])) #take order of indices into account
    vectors = np.zeros((k,1))

    for i,j in enumerate(inputs):
        matrix[indices[i]] = j
        vectors[indices[i]] += 1
    return np.divide(matrix.T,vectors.T).T


def kmeans(inputs, k, max_iter, tol):
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement
    :param inputs: inputs of data, a 2D Numpy array
    :param k: number of cluster centroids, an int
    :param max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int
    :param tol: the tolerance we determine convergence with when compared to the ratio as stated on handout
    :return: a Numpy array of k cluster centroids, one per row
    """
    # TODO
    centroids = init_centroids(k,inputs)

    for i in range(max_iter):
        indices = assign_step(inputs, centroids)
        new_centroids = update_step(inputs,indices,k)

        # dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in inputs])
        # probs = dist_sq/dist_sq.sum()
        # cumulative_probs = probs.cumsum()

        #calc differences between original and new centroids
        x_1 = np.linalg.norm(new_centroids - centroids,axis=1)
        x_2 = np.linalg.norm(centroids,axis=1)
        diff = x_1 / x_2

        if np.any(tol > diff): #if it is below tolerance threshold
            break
            # return centroids ?
        centroids = new_centroids  #replaces old centroids
    return centroids

