import torch.optim as optim
import torch
import wandb
import os

def train_vqvae(model, train_loader, val_loader, epochs=100, 
                learning_rate=1e-3, device='cuda', save_path='./models',
                project_name="vqvae-training", run_name=None, sample_rate=48000):
    
    wandb.init(project=project_name, name=run_name)
    
    config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": train_loader.batch_size if hasattr(train_loader, 'batch_size') else None,
        "model_type": model.__class__.__name__,
        "device": device,
        "data_type": "audio_waveform",
        "sample_rate": sample_rate
    }
    wandb.config.update(config)
    
    os.makedirs(save_path, exist_ok=True)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    recon_losses = []
    vq_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_vq_loss = 0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            x_recon, loss, vq_loss, recon_loss, _ = model(data)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_vq_loss += vq_loss.item()
            
            if batch_idx % 100 == 0:
                wandb.log({
                    "batch": batch_idx + epoch * len(train_loader),
                    "batch_loss": loss.item(),
                    "batch_recon_loss": recon_loss.item(),
                    "batch_vq_loss": vq_loss.item()
                })

                if epoch % 5 == 0 and isinstance(data, torch.Tensor):
                    audio_samples = []
                    for i in range(min(4, data.shape[0])):
                        original_audio = data[i].cpu().numpy()
                        recon_audio = x_recon[i].detach().cpu().numpy()

                        if original_audio.max() > 1.0 or original_audio.min() < -1.0:
                            original_audio = original_audio / max(abs(original_audio.max()), abs(original_audio.min()))

                        if recon_audio.max() > 1.0 or recon_audio.min() < -1.0:
                            recon_audio = recon_audio / max(abs(recon_audio.max()), abs(recon_audio.min()))


                        audio_samples.append(wandb.Audio(original_audio, sample_rate=sample_rate, caption=f"Original {i}"))
                        audio_samples.append(wandb.Audio(recon_audio, sample_rate=sample_rate, caption=f"Reconstruction {i}"))

                        wandb.log({"audio_examples": audio_samples})
            
        avg_loss = epoch_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_vq_loss = epoch_vq_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        recon_losses.append(avg_recon_loss)
        vq_losses.append(avg_vq_loss)
        
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_vq_loss = 0
            
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                x_recon, loss, vq_loss, recon_loss, _ = model(data)
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_vq_loss += vq_loss.item()
            
        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        avg_val_vq_loss = val_vq_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
            "recon_loss": avg_recon_loss,
            "vq_loss": avg_vq_loss,
            "val_recon_loss": avg_val_recon_loss,
            "val_vq_loss": avg_val_vq_loss
        })
        
            
        print(f"Epoch [{epoch+1}/{epochs}], Train loss: {avg_loss:.4f}, Val loss: {avg_val_loss:.4f}, Recon loss: {avg_recon_loss:.4f}, VQ loss: {avg_vq_loss:.4f}")
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(save_path, f'vqvae_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            
            wandb.save(checkpoint_path)
    
    wandb.finish()
            
    return model, train_losses, recon_losses, vq_losses, val_losses
