#include <iostream>

struct Panel {
    Panel()
    : head_(NULL),
    tail_(NULL) { }

    ~Panel() {
        Node *current = head_;
        while(current) {
            Node *temp = current;
            current = current->next_;
            delete temp;
        }
    }

    void create_panel(double data)
    {

        Node *new_panel = new Node;
        new_panel->data_ = data;
        new_panel->next_ = NULL;
        
        if(head_ == NULL) {  
            head_ = new_panel;  
            tail_ = new_panel;  
         }  
        else {  
            tail_->next_ = new_panel;  
            tail_ = new_panel;  
        }       
    } 
    
    int panels() {
        int panels = 0;
        Node *current;
	    for(current = head_; current != NULL; current = current->next_) {
            panels++;
        }	
        return panels;
    }

    void display() {
        Node *current = head_;
        if(head_ == NULL) {    
            return;  
        } 
        while(current != NULL) {
            std::cout<<current->data_<<", ";
            current = current->next_;
        }
    }

private:
    struct Node {
        double data_;
        Node* next_;
    };
    Node *head_;
    Node *tail_;
};


int main(int argc, const char * argv[]) {
    /*int data_size = 13;
    double data[13] = { 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
        2.1, 2.2, 2.3, 2.4};
    
    int initial = 10;
    for (int i = 0; i < data_size; i++) {
        buffer.append(data[i]);
    }*/
    Panel p;
    p.create_panel(1.1);
    p.create_panel(1.2);
    p.create_panel(1.3);
    p.create_panel(1.4);

    p.display();
       
    return 0;
}