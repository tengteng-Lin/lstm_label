import model

EPOCHS = 10
model1, (train_x, train_y), (test_x, test_y) = model.create_model()
# train model
model1.fit(train_x, train_y,batch_size=64,epochs=EPOCHS, validation_data=[test_x, test_y])
model1.save('without_crf.h5')
