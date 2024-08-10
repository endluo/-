import torch
from torch import nn
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, random_split
import warnings
import pandas as pd
from sklearn.metrics import accuracy_score
# 忽略所有的UserWarning
warnings.filterwarnings('ignore', category=UserWarning)

class CSVLoader:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.texts = []
        self.labels = []
        self.label_mapping = {'anger': 1, 'fear': 2, 'joy': 3, 'sadness': 4}
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.csv_file, nrows=100)
        self.texts = df['content'].tolist()
        self.labels = df['sentiment'].map(self.label_mapping).tolist()

    def get_texts_and_labels(self):
        return self.texts, self.labels

class DistillationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx] - 1, dtype=torch.long)  # Convert labels to 0-based index
        }

class DistillationModel(nn.Module):
    def __init__(self, teacher_model, student_model):
        super(DistillationModel, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model

    def forward(self, input_ids, attention_mask, labels=None, temperature=2.0, alpha=0.5):
        with torch.no_grad():
            teacher_logits = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        student_logits = self.student_model(input_ids=input_ids, attention_mask=attention_mask).logits

        teacher_probs = nn.functional.softmax(teacher_logits / temperature, dim=-1)
        student_probs = nn.functional.log_softmax(student_logits / temperature, dim=-1)

        distillation_loss = nn.functional.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
        ce_loss = nn.functional.cross_entropy(student_logits, labels)
        
        loss = alpha * ce_loss + (1 - alpha) * distillation_loss
        return loss

def train_distillation_model(train_dataloader, model, optimizer, scheduler, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

def evaluate_model(dataloader, model, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def print_model_parameters(model, model_name):
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} has {num_params:,} parameters")

if __name__ == "__main__":
    csv_loader = CSVLoader('/content/eng_dataset.csv')  
    texts, labels = csv_loader.get_texts_and_labels()
    
    # 切分数据集
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = DistillationDataset(texts, labels, tokenizer)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 加载模型
    teacher_model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=4)
    student_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

    # 打印模型参数量
    print_model_parameters(teacher_model, "Teacher Model")
    print_model_parameters(student_model, "Student Model")

    # 定义 tokenizer 和 optimizer
    optimizer = AdamW(student_model.parameters(), lr=5e-5)

    # 定义调度器
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # 总共训练步骤
    total_steps = len(train_dataloader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 初始化蒸馏模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distillation_model = DistillationModel(teacher_model, student_model).to(device)

    # 训练模型
    train_distillation_model(train_dataloader, distillation_model, optimizer, scheduler, device)

    # 保存学生模型的权重
    torch.save(student_model.state_dict(), "student_model.pth")

    # 加载学生模型的权重
    student_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    student_model.load_state_dict(torch.load("student_model.pth"))
    student_model.eval()  # 切换到评估模式

    # 计算训练集和测试集的准确率
    teacher_model.eval()
    train_accuracy_teacher = evaluate_model(train_dataloader, teacher_model, device)
    test_accuracy_teacher = evaluate_model(test_dataloader, teacher_model, device)
    train_accuracy_student = evaluate_model(train_dataloader, student_model, device)
    test_accuracy_student = evaluate_model(test_dataloader, student_model, device)

    print(f"Teacher Model - Training Accuracy: {train_accuracy_teacher:.4f}")
    print(f"Teacher Model - Testing Accuracy: {test_accuracy_teacher:.4f}")
    print(f"Student Model - Training Accuracy: {train_accuracy_student:.4f}")
    print(f"Student Model - Testing Accuracy: {test_accuracy_student:.4f}")
