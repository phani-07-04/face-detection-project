import os
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.neighbors import KNeighborsClassifier
import joblib

accused_dir = r'C:\Program Files\accused'
normal_citizen_dir = r'C:\Program Files\normal citizen'
model_path = 'knn_facenet_model.joblib'

labels = {'accused': 0, 'normal_citizen': 1}
label_names = {0: 'Accused', 1: 'Normal Citizen'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_embeddings(image_path):
    img = Image.open(image_path).convert('RGB')
    face = mtcnn(img)
    if face is not None:
        face_embedding = resnet(face.unsqueeze(0).to(device))
        return face_embedding.detach().cpu().numpy()[0]
    return None

if not os.path.exists(model_path):
    print("Training KNN classifier...")

    embeddings = []
    face_labels = []

    for img in os.listdir(accused_dir):
        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(accused_dir, img)
            embedding = extract_embeddings(path)
            if embedding is not None:
                embeddings.append(embedding)
                face_labels.append(labels['accused'])

    for img in os.listdir(normal_citizen_dir):
        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(normal_citizen_dir, img)
            embedding = extract_embeddings(path)
            if embedding is not None:
                embeddings.append(embedding)
                face_labels.append(labels['normal_citizen'])

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(embeddings, face_labels)

    joblib.dump(knn, model_path)
    print("Model trained and saved as 'knn_facenet_model.joblib'.")

else:
    print("Loading saved model...")
    knn = joblib.load(model_path)

print("Starting webcam for live recognition...")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb_frame)
    face = mtcnn(img_pil)

    if face is not None:
        face_embedding = resnet(face.unsqueeze(0).to(device))
        embedding_np = face_embedding.detach().cpu().numpy()

        pred = knn.predict(embedding_np)
        prob = knn.predict_proba(embedding_np)
        confidence = np.max(prob) * 100

        label = label_names.get(pred[0], "Unknown")
        color = (0, 0, 255) if pred[0] == 0 else (0, 255, 0)

        cv2.putText(frame, f"{label} ({confidence:.2f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Face Recognition - FaceNet + KNN", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
