package com.sparrowrecsys.online;

import com.sparrowrecsys.online.datamanager.DataManager;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.DefaultServlet;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.eclipse.jetty.util.resource.Resource;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URL;


/**
 * Recsys Server, end point of online recommendation service
 */
public class RecSysServer {
    // 主函数，创建推荐服务器并运行
    public static void main(String[] args) throws Exception {
        new RecSysServer().run();
    }
    // recsys server port number
    private static final int DEFAULT_PORT = 6010;

    public void run() throws Exception{
        int port = DEFAULT_PORT;
        try {
            port = Integer.parseInt(System.getenv("PORT"));
        } catch (NumberFormatException ignored) {}

        // set ip and port number
        InetSocketAddress inetAddress = new InetSocketAddress("0.0.0.0", port);
        Server server = new Server(inetAddress);

        // get index.html path
        URL webRootLocation = this.getClass().getResource("/webroot/index.html");
        if(webRootLocation == null)
        {
            throw new IllegalStateException("Unable to determine webroot URL location");
        }

        // set index.html as the root page
        URI webRootUri = URI.create(webRootLocation.toURI().toASCIIString().replaceFirst("/index.html$", "/"));
        System.out.printf("Web Root URI: %s%n", webRootUri.getPath());

        //
    }
}
