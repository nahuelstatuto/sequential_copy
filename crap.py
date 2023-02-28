def create_copy_model(d, 
                      n_classes,
                      lr,
                      optimizer='Adam',
                      input_shape=64,
                      hidden_layers=[64, 32, 10],
                      loss_name='self',
                      activation='relu'):
    
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss, loss_function = define_loss(loss_name, d)   
    copy_model = fn.CustomCopyModel(input_dim=d, 
                                    hidden_layers=hidden_layers,
                                    output_dim=n_classes,
                                    activation=activation)
    
    copy_model.build(input_shape=(input_shape,d))
    copy_model.compile(optimizer=opt, loss=loss)

    return copy_model, opt, loss_function