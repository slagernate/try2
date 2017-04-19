import tensorflow as tf

#helper functions
def bernoulli_sample(x):
    """
    return tensor with element yi turned "on"
    with probability xi
    """
    return tf.ceil(x - tf.random_uniform(tf.shape(x), minval = 0, maxval=1))

#assumes session is already running
def get_tensor_value(tensor_name, collection_name):
    c = tf.get_collection(collection_name)
    for item in c:
        if item.name == tensor_name:
            print("Restored " + item.name)
            return item
    else:
        print("NO ITEMS IN COLLECTION. FIX IT!")
        return None
    print("TENSOR NOT FOUND IN COLLECTION. SORRY FOR ALL CAPS BUT YOU NEED TO FIX THIS")
    return None



