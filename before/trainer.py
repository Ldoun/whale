class Trainer():
    def __init__(self,args, writer, model, device, optimizer, criterion, train_dataloader, valid_dataloader):
        self.args = args
        self.writer = writer
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.fold = 0
     
    def train_loop(self, epoch):
        self.model.train()
        itr_start_time = time.time()
        n_iters = len(self.train_dataloader)
        
        print('befor dataloading')
        for step, batch in enumerate(self.train_dataloader, start=1):
            if step == 1:
                print('data loaded')
                
            self.optimizer.zero_grad()
            
            #image = batch['image']
            #label = batch['label']

            image, label = batch
            
            prediction = self.model(image)
            
            loss = self.criterion(prediction, label)
            loss.backward()
            
            xm.optimizer_step(self.optimizer)
            #optimizer.step()
            
            if step % self.args.log_every == 0:
                elapsed = time.time() - itr_start_time
                xm.add_step_closure(
                    train_logging,      
                    args=(self.writer,epoch, self.args.total_epoch,step,n_iters,elapsed, loss.item()),
                )
                itr_start_time = time.time()
                #print('[xla:{}]({}/{}) Loss={:.5f} Time={}'.format(xm.get_ordinal(), step,n_iters,loss.item(), time.asctime()), flush=True)
                
    def validation_loop(self, epoch):
        self.model.eval()
        
        correct, total = 0,0
        n_iters = len(self.valid_dataloader)
        
        with torch.no_grad():
            for step, batch in enumerate(self.valid_dataloader, start=1):
                #image = batch['image']
                #label = batch['label']
                
                image, label = batch
                
                prediction = self.model(image)
                
                loss = self.criterion(prediction, label)
                
                _, predicted = torch.max(prediction.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            
                xm.add_step_closure(
                    valid_logging,
                    args=(self.writer, epoch, self.args.total_epoch, step, n_iters, loss.item(), correct, total),
                )
                #print('valid [xla:{}]({}) Loss={:.5f} Time={}'.format(xm.get_ordinal(), step, loss.item(), time.asctime()), flush=True)
                    
                correct, total = 0,0
                
    def save_model(self, epoch):
        dict_for_infer = {
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict(),
            #"scaler": scheduler.state_dict(),
            #"amp": amp.state_dict(),
            "batch_size": self.args.batch_size,
            "epochs": self.args.total_epoch,
            "learning_rate": self.args.lr,
        }
        
        os.makedirs(self.args.ckt_folder, exist_ok=True)
        save_dir = os.path.join(self.args.ckt_folder, f"{self.args.model_name}-checkpoint_{self.fold}fold_{epoch}epoch")
        
        xm.save(dict_for_infer, save_dir)
        
        
    def train(self):
        start_epoch = 0
        
        for epoch in range(start_epoch, self.args.total_epoch + 1):
            self.train_loop(epoch)
            print(f'validation start{epoch}')
            self.validation_loop(epoch)          
            if epoch % self.args.save_every == 0:
                if xm.is_master_ordinal():
                    xm.add_step_closure(
                        self.save_model,
                        args = (epoch,),   
                    )
                    
                    print('saved')