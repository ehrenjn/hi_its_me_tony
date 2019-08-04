#model basically copied and pasted https://adventuresinmachinelearning.com/word2vec-keras-tutorial/
    #changed the rmsprop optimizer to adam
    #changed dot product to cosine similarity

'''
Questions:
    WHY DO I NEED TO RESHAPE TO DO DOT PRODUCT??
'''


from keras import layers, models
from consts import NUM_INDEXED_WORDS, WORD2VEC_VEC_SIZE


def create_model():

    target_input = layers.Input((1,)) #going to be tossing in 1 word at a time
    context_input = layers.Input((1,))

    #create a layer that takes a 1 hot index as input and outputs an embedding
    embedding = layers.Embedding(
        NUM_INDEXED_WORDS, #one hot encoding dimentionality 
        WORD2VEC_VEC_SIZE, #embedding size
        #input_length = 1, #input length is 1 since we're not doing any sequences
        name = 'embedding' #name it so I can access it easily from the model later
    )

    #make a new layer who's input is target_input and output is one of our embedding layers
    target = embedding(target_input) 
    context = embedding(context_input) 

    #plug target into a reshape layer that output goes from [1,2,3] to [[1], [2], [3]])
    target = layers.Reshape((WORD2VEC_VEC_SIZE, 1))(target)
    context = layers.Reshape((WORD2VEC_VEC_SIZE, 1))(context)



    # setup a cosine similarity operation which will be output in a secondary model
    #similarity = merge([target, context], mode = 'cos', dot_axes = 0)


    #combine both embeddings
    dot_product = layers.dot( #cosine similarity is just a normalized dot product
        [target, context], 
        axes = 1, #specify which axis of input to do the dot product over
        normalize = True #normalize so that this dot product becomes a cosine similarity
    )

    #reshape back to a 1D vector
    dot_product = layers.Reshape((1,))(dot_product)

    #put into sigmoid just to scale the cosine properly
    output = layers.Dense(1, activation = 'sigmoid')(dot_product)

    #create and compile model
    model = models.Model(input = [target_input, context_input], output = output)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')


    # create a secondary validation model to run our similarity checks during training
    #validation_model = Model(input = [input_target, input_context], output = similarity)