from MLModel.echonet_ef_model import infer
ef = infer('/Users/abhiramasonny/Developer/ScienceFair 2024-2025/dataset/ef_model.pt', 'EchoNet-Dynamic/videos/0X211D307253ACBEE7.avi')
print(f"Predicted EF: {ef:.1f} %")