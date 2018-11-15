package top.deepzone.rjzhou.dlib_bench;

import android.os.Handler;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import static java.security.AccessController.getContext;

public class MainActivity extends AppCompatActivity {



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        findViewById(R.id.button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                findViewById(R.id.button).setEnabled(false);

                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        long eclipsed = Test.convolutionTest();
                        System.out.println("-----------------------------------CONV " + String.valueOf(eclipsed));
                        Message message = new Message();
                        message.what = 1;
                        message.arg1 = (int)eclipsed;
                        mHandler.sendMessage(message);

                        eclipsed = Test.alexnetTest();
                        System.out.println("-----------------------------------ALEX " + String.valueOf(eclipsed));
                        message = new Message();
                        message.what = 2;
                        message.arg1 = (int)eclipsed;
                        mHandler.sendMessage(message);

                        eclipsed = Test.resnetTest();
                        System.out.println("-----------------------------------RESNET " + String.valueOf(eclipsed));
                        message = new Message();
                        message.what = 3;
                        message.arg1 = (int)eclipsed;
                        mHandler.sendMessage(message);

                        eclipsed = Test.inceptionTest();
                        System.out.println("-----------------------------------INCEPT " + String.valueOf(eclipsed));
                        message = new Message();
                        message.what = 4;
                        message.arg1 = (int)eclipsed;
                        mHandler.sendMessage(message);
                    }
                }).start();
            }
        });
    }

    private Handler mHandler = new Handler(){
        @Override public void handleMessage(Message msg) {
            super.handleMessage(msg);
            if (msg.what == 1) {
                TextView t = findViewById(R.id.conv_time);
                t.setText(String.valueOf(msg.arg1));
            }
            else if (msg.what == 2) {
                TextView t = findViewById(R.id.alex_time);
                t.setText(String.valueOf(msg.arg1));
            }
            else if (msg.what == 3) {
                TextView t = findViewById(R.id.res_time);
                t.setText(String.valueOf(msg.arg1));
            }
            else if (msg.what == 4) {
                TextView t = findViewById(R.id.incept_time);
                t.setText(String.valueOf(msg.arg1));

                findViewById(R.id.button).setEnabled(true);
            }
        }
    };
}
