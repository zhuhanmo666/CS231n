def loss(self, X, y=None):
    """Compute loss and gradient for the fully connected net.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
        scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
        names to gradients of the loss with respect to those parameters.
    """
    X = X.astype(self.dtype)
    mode = "test" if y is None else "train"

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.

    if self.normalization == "batchnorm":
        for bn_param in self.bn_params:
            bn_param["mode"] = mode
    ############################################################################
    # TODO: Implement the forward pass for the fully connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    out = X
    caches = {}

    for i in range(self.num_layers - 1):
        W = self.params.get(f'W{i + 1}', None)
        b = self.params.get(f'b{i + 1}', None)

        if W is None or b is None:
            print(f"Skipping layer {i + 1} because W or b is None")
            continue
        out, fc_cache = affine_forward(out, W, b)

        if self.normalization == "batchnorm":
            gamma = self.params[f"gamma{i}"]
            beta = self.params[f"beta{i}"]
            if gamma is not None and beta is not None:
                out, bn_cache = batchnorm_forward(out, gamma, beta, self.bn_params[i])
                caches[f'bn{i + 1}'] = bn_cache
        out, relu_cache = relu_forward(out)
        caches[f'affine{i + 1}'] = fc_cache
        caches[f'relu{i + 1}'] = relu_cache

    W = self.params.get(f'W{self.num_layers}', None)
    b = self.params.get(f'b{self.num_layers}', None)
    if W is not None and b is not None:
        scores, final_cache = affine_forward(out, W, b)
        caches[f'affine{self.num_layers}'] = final_cache
    else:
        scores = out
        ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early.
    if mode == "test":
        return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch/layer normalization, you don't need to regularize the   #
    # scale and shift parameters.                                              #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    # L2 regularization loss
    for i in range(1, self.num_layers + 1):
        W = self.params[f"W{i}"]
        loss += 0.5 * self.reg * np.sum(W * W)

    grads = {}

    # last layer backward
    dout, dW, db = affine_backward(dscores, cache_last)
    grads[f"W{self.num_layers}"] = dW + self.reg * self.params[f"W{self.num_layers}"]
    grads[f"b{self.num_layers}"] = db

    # hidden layers backward
    for i in reversed(range(1, self.num_layers)):
        fc_cache, bn_cache, relu_cache, do_cache = caches[i]

        if self.use_dropout:
            dout = dropout_backward(dout, do_cache)

        dout = relu_backward(dout, relu_cache)

        if self.normalization == "batchnorm":
            dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
            grads[f"gamma{i}"] = dgamma
            grads[f"beta{i}"] = dbeta

        dout, dW, db = affine_backward(dout, fc_cache)
        grads[f"W{i}"] = dW + self.reg * self.params[f"W{i}"]
        grads[f"b{i}"] = db
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads